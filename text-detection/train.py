import torch
import torch.nn as nn
import argparse
import os
import os.path as osp
from glob import glob
from models.utils import evaluate_micro, get_iou
from hydra import compose, initialize
from models.fast import FAST
from models.loss.ohem import get_ohem_masks
from models.loss.bce_loss import BCELoss
from models.loss.dice_loss import DiceLoss
from models.loss.auf_loss import AUFLoss
from models.schedulers import PretrainingScheduler
from tqdm import tqdm
import time
from torch.utils.data import DataLoader
from loader_new import FAST_IC15, DataLoaderIterator
import wandb
from dali_loader import get_loader

# wandb.login()


def get_run_id(cfg):
    prev_runs = glob(f"checkpoints/{cfg}*")
    if len(prev_runs) == 0:
        prev_run_id = -1
    else:
        prev_run_ids = [int(run[-3:]) for run in prev_runs]
        prev_run_id = sorted(prev_run_ids)[-1]
    return f"{cfg}_{str(prev_run_id + 1).zfill(3)}"


def calculate_total_loss(kernel_loss, text_loss):
    return kernel_loss + text_loss / 3


def main(cfg, args):
    # def main():
    # with initialize(version_base=None, config_path="config", job_name="test"):
    #     cfg = compose(config_name="ic15_auf")
    wandb.init(project="text-detection", config=dict(cfg), resume="allow")

    batch_size = cfg.data.batch_size
    train_loader = get_loader(
        cfg.data.dataset, "train", 640, batch_size=batch_size, num_threads=1
    )
    val_loader = get_loader(
        cfg.data.dataset, "val", 736, batch_size=batch_size, num_threads=1
    )
    # if cfg.data.dataset == "ic15":
    # train_dset = FAST_IC15("train", short_size=640)
    # train_loader = DataLoaderIterator(
    #     train_dset, shuffle=True, batch_size=batch_size
    # )
    # val_dset = FAST_IC15("val", short_size=736)
    # val_loader = DataLoaderIterator(val_dset, shuffle=True, batch_size=batch_size)

    # elif cfg.data.dataset == "synthtext":
    #     train_loader, val_loader = get_synthtext_loaders(batch_size=batch_size)

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
    if cfg.model.kernel_loss.loss_fn == "bce_loss":
        kernel_loss_fn = BCELoss()
    elif cfg.model.kernel_loss.loss_fn == "dice_loss":
        kernel_loss_fn = DiceLoss()
    elif cfg.model.kernel_loss.loss_fn == "auf_loss":
        kernel_loss_fn = AUFLoss(gamma=cfg.model.kernel_loss.gamma)
    kernel_ohem = cfg.model.kernel_loss.ohem

    # text loss
    if cfg.model.text_loss.loss_fn == "bce_loss":
        text_loss_fn = BCELoss()
    elif cfg.model.text_loss.loss_fn == "dice_loss":
        text_loss_fn = DiceLoss()
    elif cfg.model.text_loss.loss_fn == "auf_loss":
        text_loss_fn = AUFLoss(gamma=cfg.model.text_loss.gamma)
    text_ohem = cfg.model.text_loss.ohem

    # optimizer
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.train.lr,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
            fused=True,
        )

    # if cfg.train.schedule == "cosine_warmup":
    #     scheduler = PretrainingScheduler(
    #         optimizer,
    #         total_steps=num_iterations,
    #         warmup_steps=cfg.train.warmup_iterations,
    #         eta_min=cfg.train.min_lr,
    #     )

    # learning rate scheduler
    if not cfg.train.warmup:
        warmup_iterations = 0
    else:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            total_iters=cfg.train.warmup_iterations,
        )
        warmup_iterations = cfg.train.warmup_iterations

    if cfg.train.schedule == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, power=0.9, total_iters=num_iterations - warmup_iterations
        )
    if cfg.train.warmup:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler], [warmup_iterations]
        )

    # load checkpoint
    iteration = 0
    best_val_loss = float("inf")
    if cfg.train.resume:
        checkpoint = torch.load(
            osp.join("checkpoints", cfg.train.resume, "recent.pth"), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        iteration = checkpoint["iteration"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint["best_val_loss"]

        checkpoint_path = osp.join("checkpoints", cfg.train.resume)
        print(f"Resuming at iteration {iteration}")
    else:
        if cfg.train.checkpoint:
            checkpoint = torch.load(
                osp.join("checkpoints", cfg.train.checkpoint), weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_loss = checkpoint["best_val_loss"]
            print(f"Loaded checkpoint {cfg.train.checkpoint}")

        run_id = get_run_id("auf_loss")
        checkpoint_path = osp.join("checkpoints", run_id)
        os.makedirs(checkpoint_path)
    print(f"Checkpoint path: {checkpoint_path}")

    # train for num_iterations
    while iteration < num_iterations:
        # train loop
        model.train()
        torch.cuda.empty_cache()
        train_kernel_running_loss = 0.0
        train_text_running_loss = 0.0
        train_running_loss = 0.0

        for _ in range(train_eval_interval):
            batch = next(train_loader)[0]
            images = batch["images"]
            gt_kernels = batch["gt_kernels"]
            gt_texts = batch["gt_texts"]
            training_masks = batch["training_masks"]

            optimizer.zero_grad()

            preds = model(images)
            dilated_preds = torch.nn.functional.max_pool2d(
                preds,
                kernel_size=9,
                stride=1,
                padding=4,
            )

            if kernel_ohem:
                ohem_kernel_mask = get_ohem_masks(preds, gt_kernels, training_masks)
                loss_kernel = kernel_loss_fn(preds, gt_kernels, ohem_kernel_mask)
            else:
                loss_kernel = kernel_loss_fn(preds, gt_kernels, training_masks)
            if text_ohem:
                ohem_text_mask = get_ohem_masks(dilated_preds, gt_texts, training_masks)
                loss_text = text_loss_fn(dilated_preds, gt_texts, ohem_text_mask)
            else:
                loss_text = text_loss_fn(dilated_preds, gt_texts, training_masks)
            loss = calculate_total_loss(loss_kernel, loss_text)
            loss.backward()

            optimizer.step()
            scheduler.step()

            train_kernel_running_loss += loss_kernel.item()
            train_text_running_loss += loss_text.item()
            train_running_loss += loss.item()
        train_kernel_loss = train_kernel_running_loss / train_eval_interval
        train_text_loss = train_text_running_loss / train_eval_interval
        train_loss = train_running_loss / train_eval_interval

        # validation loop
        model.eval()
        torch.cuda.empty_cache()
        val_kernel_running_loss = 0.0
        val_text_running_loss = 0.0
        val_running_loss = 0.0
        val_running_kernel_iou = 0.0
        val_running_text_iou = 0.0

        with torch.no_grad():
            for _ in range(val_eval_interval):
                batch = next(val_loader)[0]
                images = batch["images"]
                gt_kernels = batch["gt_kernels"]
                gt_texts = batch["gt_texts"]
                training_masks = batch["training_masks"]

                preds = model(images)
                dilated_preds = torch.nn.functional.max_pool2d(
                    preds,
                    kernel_size=9,
                    stride=1,
                    padding=4,
                )

                if kernel_ohem:
                    ohem_kernel_mask = get_ohem_masks(preds, gt_kernels, training_masks)
                    loss_kernel = kernel_loss_fn(preds, gt_kernels, ohem_kernel_mask)
                else:
                    loss_kernel = kernel_loss_fn(preds, gt_kernels, training_masks)
                if text_ohem:
                    ohem_text_mask = get_ohem_masks(
                        dilated_preds, gt_texts, training_masks
                    )
                    loss_text = text_loss_fn(dilated_preds, gt_texts, ohem_text_mask)
                else:
                    loss_text = text_loss_fn(dilated_preds, gt_texts, training_masks)
                    loss = calculate_total_loss(loss_kernel, loss_text)

                val_kernel_running_loss += loss_kernel.item()
                val_text_running_loss += loss_text.item()
                val_running_loss += loss.item()

                val_running_kernel_iou += get_iou(preds, gt_kernels, training_masks)
                val_running_text_iou += get_iou(dilated_preds, gt_texts, training_masks)

            val_kernel_loss = val_kernel_running_loss / val_eval_interval
            val_text_loss = val_text_running_loss / val_eval_interval
            val_loss = val_running_loss / val_eval_interval
            val_kernel_iou = val_running_kernel_iou / val_eval_interval
            val_text_iou = val_running_text_iou / val_eval_interval

        iteration += train_eval_interval

        # log to wandb
        print(
            f"[Iter {iteration}] | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | IOU(kernel/text): {val_kernel_iou:.4f}/{val_text_iou:.4f} | lr: {scheduler.get_last_lr()[0]:.7f}"
        )
        log = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "kernel_loss_train": train_kernel_loss,
            "kernel_loss_val": val_kernel_loss,
            "text_loss_train": train_text_loss,
            "text_loss_val": val_text_loss,
            "kernel_iou": val_kernel_iou,
            "text_iou": val_text_iou,
        }

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
        if iteration % cfg.train.save_interval == 0:
            precision, recall, f1 = evaluate_micro(model, val_loader, val_eval_interval)
            print(f"precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f}")
            log["precision"] = precision
            log["recall"] = recall
            log["f1"] = f1

        wandb.log(log)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--checkpoint", nargs="?", type=str, default=None)
    parser.add_argument("--resume", nargs="?", type=str, default=None)
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config", job_name="test"):
        cfg = compose(config_name=args.config)

    # sweep_config = {
    #     "method": "grid",
    #     "metric": {"name": "f1", "goal": "minimize"},
    #     "parameters": {
    #         "auf_gamma": {"values": [0.9, 0.1, 0.5, 0.7, 0.3]},
    #     },
    # }
    # sweep_id = wandb.sweep(sweep=sweep_config, project="text-detection")
    # wandb.agent(sweep_id, function=main)

    main(cfg, args)
    # main()
