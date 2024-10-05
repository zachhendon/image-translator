import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from hydra import compose, initialize
from data.dataloader import get_loaders
from models.fast import FAST
from models.fast import get_run_id
from models.loss.dice_loss import DiceLoss
from models.loss.ufl_asym_loss import UFLAsymLoss
from models.loss.ufl_sym_loss import UFLSymLoss
from models.loss.balanced_bce_loss import BalancedBCELoss
from models.loss.ohem import get_ohem_masks
import os
import os.path as osp
from tqdm import tqdm
from models.utils import evaluate


def main(cfg, args):
    # dsets
    if cfg.data.type == "icdar2015":
        train_loader, val_loader = get_loaders(
            cfg.data.datadir, cfg.data.batch_size, train=True
        )

    # model
    if cfg.model.type == "fast":
        model = FAST()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()

    # loss
    if cfg.model.loss == "dice_loss":
        loss_fn = DiceLoss()
    elif cfg.model.loss == "ufl_asym_loss":
        loss_fn = UFLAsymLoss()
    elif cfg.model.loss == "ufl_sym_loss":
        loss_fn = UFLSymLoss()
    elif cfg.model.loss == "bce_loss":
        loss_fn = nn.BCELoss()
    elif cfg.model.loss == "balanced_bce_loss":
        loss_fn = BalancedBCELoss()

    # optimizer
    if cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    elif cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr)

    if cfg.train.schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.train.epochs
        )
    elif cfg.train.schedule == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, cfg.train.epochs)

    # load checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(
            osp.join("checkpoints", args.resume, "recent.pth"), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        checkpoint_path = osp.join("checkpoints", args.resume)
        print(f"Resuming at epoch {start_epoch + 1}")
    else:
        if args.checkpoint:
            checkpoint = torch.load(
                osp.join("checkpoints", args.checkpoint), weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint {args.checkpoint}")

        run_id = get_run_id(args.config)
        checkpoint_path = osp.join("checkpoints", run_id)
        os.makedirs(checkpoint_path)

    # tensorboard
    print(f"Checkpoint path: {checkpoint_path}")
    writer = SummaryWriter(log_dir=checkpoint_path)

    # train loop
    dice_loss = DiceLoss()
    best_val_loss = float("inf")
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        train_running_loss = 0.0
        train_dataset_size = 0
        for images, (gt_kernels, gt_texts, kernel_masks, text_masks, _) in train_loader:
            if use_cuda:
                images = images.to(dtype=torch.float32, device="cuda")
                gt_kernels = gt_kernels.to(dtype=torch.float32, device="cuda")
                gt_texts = gt_texts.to(dtype=torch.float32, device="cuda")
                kernel_masks = kernel_masks.to(dtype=torch.float32, device="cuda")
                text_masks = text_masks.to(dtype=torch.float32, device="cuda")

            optimizer.zero_grad()

            preds = model(images)
            dilated_preds = torch.nn.functional.max_pool2d(
                preds, kernel_size=9, stride=1, padding=4
            )

            kernel_masks = get_ohem_masks(preds, gt_kernels, kernel_masks, ratio=3.0)
            text_masks = get_ohem_masks(dilated_preds, gt_texts, text_masks, ratio=3.0)
            loss_kernel = loss_fn(preds, gt_kernels, kernel_masks)
            # loss_text = dice_loss(dilated_preds, gt_texts, torch.ones_like(text_masks))
            loss_text = loss_fn(dilated_preds, gt_texts, text_masks)
            loss = loss_kernel + 0.5 * loss_text
            loss.backward()

            optimizer.step()

            batch_size = images.size(0)
            train_running_loss += loss.item() * batch_size
            train_dataset_size += batch_size

        train_loss = train_running_loss / train_dataset_size
        scheduler.step()

        # validation loop
        model.eval()
        val_running_loss = 0.0
        val_dataset_size = 0
        with torch.no_grad():
            for images, (gt_kernels, gt_texts, kernel_masks, text_masks, _) in val_loader:
                if use_cuda:
                    images = images.to(dtype=torch.float32, device="cuda")
                    gt_kernels = gt_kernels.to(dtype=torch.float32, device="cuda")
                    gt_texts = gt_texts.to(dtype=torch.float32, device="cuda")
                    kernel_masks = kernel_masks.to(dtype=torch.float32, device="cuda")
                    text_masks = text_masks.to(dtype=torch.float32, device="cuda")

                preds = model(images)
                dilated_preds = torch.nn.functional.max_pool2d(
                    preds, kernel_size=9, stride=1, padding=4
                )

                # kernel_masks = get_ohem_masks(preds, gt_kernels, kernel_masks, ratio=3.0)
                # text_masks = get_ohem_masks(dilated_preds, gt_texts, text_masks, ratio=3.0)
                loss_kernel = loss_fn(preds, gt_kernels, kernel_masks)
                # loss_text = dice_loss(dilated_preds, gt_texts, torch.ones_like(text_masks))
                loss_text = loss_fn(dilated_preds, gt_texts, text_masks)
                loss = loss_kernel + 0.5 * loss_text

                batch_size = images.size(0)
                val_running_loss += loss.item() * batch_size
                val_dataset_size += batch_size

        val_loss = val_running_loss / val_dataset_size

        # log losses to tensorboard
        print(
            f"[Epoch {epoch + 1}] | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}"
        )
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.flush()

        # evaluate every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            precision, recall, f1 = evaluate(model, val_loader)
            print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}\n")
            writer.add_scalars(
                "metrics", {"precision": precision, "recall": recall, "f1": f1}, epoch
            )

        # save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(checkpoint, osp.join(checkpoint_path, "recent.pth"))
        if epoch % cfg.train.save_interval == 0:
            torch.save(checkpoint, osp.join(checkpoint_path, f"epoch_{epoch}.pth"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, osp.join(checkpoint_path, "best.pth"))
    writer.close()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--checkpoint", nargs="?", type=str, default=None)
    parser.add_argument("--resume", nargs="?", type=str, default=None)
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config", job_name="test"):
        cfg = compose(config_name=args.config)

    main(cfg, args)
