import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import os.path as osp
import glob
from models.utils import evaluate, evaluate_micro
from hydra import compose, initialize
from loader import get_icdar2015_loaders
from models.fast import FAST
from models.loss.ohem import get_ohem_masks
from models.loss.balanced_bce_loss import BalancedBCELoss
from models.loss.dice_loss import DiceLoss
from models.schedulers import PretrainingScheduler
from tqdm import tqdm
import time


def get_run_id(cfg):
    prev_runs = glob.glob(f"checkpoints/{cfg}*")
    if len(prev_runs) == 0:
        prev_run_id = -1
    else:
        prev_run_ids = [int(run[-3:]) for run in prev_runs]
        prev_run_id = sorted(prev_run_ids)[-1]
    return f"{cfg}_{str(prev_run_id + 1).zfill(3)}"


def calculate_total_loss(kernel_loss, text_loss):
    return 3 * kernel_loss + text_loss


def main(cfg, args):
    batch_size = cfg.data.batch_size
    # train_loader, val_loader = get_loader(batch_size, cfg.data.datadir)
    train_loader, val_loader = get_icdar2015_loaders("data/processed", batch_size)

    num_iterations = cfg.train.num_iterations
    if cfg.train.train_eval_interval:
        train_eval_interval = cfg.train.train_eval_interval
    else:
        train_eval_interval = len(train_loader)
    if cfg.train.val_eval_interval:
        val_eval_interval = cfg.train.val_eval_interval
    else:
        val_eval_interval = len(val_loader)

    # model
    if cfg.model.type == "fast":
        model = FAST()
    model = model.cuda()

    # kernel loss
    if cfg.model.kernel_loss == "balanced_bce_loss":
        kernel_loss_fn = BalancedBCELoss()

    # text loss
    if cfg.model.text_loss == "dice_loss":
        text_loss_fn = DiceLoss()

    # optimizer
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    if cfg.train.schedule == "cosine_warmup":
        scheduler = PretrainingScheduler(
            optimizer,
            total_steps=num_iterations,
            warmup_steps=cfg.train.warmup_iterations,
            eta_min=cfg.train.min_lr,
        )

    # load checkpoint
    iteration = 0
    best_val_loss = float("inf")
    if args.resume:
        checkpoint = torch.load(
            osp.join("checkpoints", args.resume, "recent.pth"), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        iteration = checkpoint["iteration"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]

        checkpoint_path = osp.join("checkpoints", args.resume)
        print(f"Resuming at iteration {iteration}")
    else:
        if args.checkpoint:
            checkpoint = torch.load(
                osp.join("checkpoints", args.checkpoint), weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded checkpoint {args.checkpoint}")

        run_id = get_run_id(args.config)
        checkpoint_path = osp.join("checkpoints", run_id)
        os.makedirs(checkpoint_path)

    # tensorboard
    print(f"Checkpoint path: {checkpoint_path}")
    writer = SummaryWriter(log_dir=checkpoint_path)

    # train for num_iterations
    while iteration < num_iterations:
        # train loop
        model.train()
        train_kernel_running_loss = 0.0
        train_text_running_loss = 0.0
        train_running_loss = 0.0

        for _ in range(train_eval_interval):
            images, kernel_masks, ignore_kernel_masks, text_masks, ignore_text_masks = (
                next(train_loader)
            )

            optimizer.zero_grad()

            preds = model(images)
            dilated_preds = torch.nn.functional.max_pool2d(
                preds, kernel_size=9, stride=1, padding=4
            )

            loss_kernel = kernel_loss_fn(preds, kernel_masks, ignore_kernel_masks)
            loss_text = text_loss_fn(dilated_preds, text_masks, ignore_text_masks)
            loss = calculate_total_loss(loss_kernel, loss_text)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_kernel_running_loss += loss_kernel.item()
            train_text_running_loss += loss_text.item()
            train_running_loss += loss.item()
        train_kernel_loss = train_kernel_running_loss / (
            train_eval_interval * batch_size
        )
        train_text_loss = train_text_running_loss / (train_eval_interval * batch_size)
        train_loss = train_running_loss / (train_eval_interval * batch_size)

        # validation loop
        model.eval()
        val_kernel_running_loss = 0.0
        val_text_running_loss = 0.0
        val_running_loss = 0.0

        with torch.no_grad():
            for _ in range(val_eval_interval):
                images, kernel_masks, ignore_kernel_masks, text_masks, ignore_text_masks = (
                    next(val_loader)
                )

                preds = model(images)
                dilated_preds = torch.nn.functional.max_pool2d(
                    preds, kernel_size=9, stride=1, padding=4
                )

                loss_kernel = kernel_loss_fn(preds, kernel_masks, ignore_kernel_masks)
                loss_text = text_loss_fn(dilated_preds, text_masks, ignore_text_masks)
                loss = calculate_total_loss(loss_kernel, loss_text)

                val_kernel_running_loss += loss_kernel.item()
                val_text_running_loss += loss_text.item()
                val_running_loss += loss.item()
            val_kernel_loss = val_kernel_running_loss / (val_eval_interval * batch_size)
            val_text_loss = val_text_running_loss / (val_eval_interval * batch_size)
            val_loss = val_running_loss / (val_eval_interval * batch_size)

        iteration += train_eval_interval

        # log to tensorboard
        print(
            f"[Iter {iteration}] | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.7f}"
        )
        writer.add_scalars(
            "kernel_loss",
            {"train": train_kernel_loss, "val": val_kernel_loss},
            iteration,
        )
        writer.add_scalars(
            "text_loss", {"train": train_text_loss, "val": val_text_loss}, iteration
        )
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, iteration)
        writer.flush()

        # save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "iteration": iteration,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, osp.join(checkpoint_path, "recent.pth"))
        if int(iteration / train_eval_interval) % cfg.train.save_interval == 0:
            torch.save(checkpoint, osp.join(checkpoint_path, f"{iteration}.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, osp.join(checkpoint_path, "best.pth"))

        # evaluate model
        if int(iteration / train_eval_interval) % cfg.train.save_interval == 0:
            pass  # TODO implement evaluate functions
            # evaluate(model, val_loader, iteration, writer)
            # evaluate_micro(model, val_loader, iteration, writer)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--checkpoint", nargs="?", type=str, default=None)
    parser.add_argument("--resume", nargs="?", type=str, default=None)
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config", job_name="test"):
        cfg = compose(config_name=args.config)

    main(cfg, args)
