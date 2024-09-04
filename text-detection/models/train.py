import glob
from data.dataloader import get_loaders
from model import DBNet
from torch.utils.tensorboard import SummaryWriter
from torch.nn import BCELoss, L1Loss
import torch
import os
import sys

sys.path.append(os.path.abspath(".."))


bin_loss_fn = BCELoss()
prob_loss_fn = BCELoss()
thresh_loss_fn = L1Loss()


def db_loss(pred, gt, train=True):
    if train:
        bin_loss = bin_loss_fn(pred["bin_map"], gt["bin_map"])
        prob_loss = prob_loss_fn(pred["prob_map"], gt["bin_map"])
        thresh_loss = thresh_loss_fn(pred["thresh_map"], gt["thresh_map"])
        loss = prob_loss + bin_loss + 10 * thresh_loss
    else:
        loss = 0
    return loss


def get_run_id():
    prev_runs = glob.glob("../runs/*")
    if len(prev_runs) == 0:
        prev_run_id = -1
    else:
        prev_run_ids = [int(run[-3:]) for run in prev_runs]
        prev_run_id = sorted(prev_run_ids)[-1]
    return f"dbnet_{str(prev_run_id + 1).zfill(3)}"


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn):
    model.train()

    running_loss = 0
    dataset_size = 0

    for images, maps in train_loader:
        optimizer.zero_grad()
        batch_size = len(images)

        images = images.to(device="cuda", dtype=torch.float32)
        images = images.permute(0, 3, 1, 2)
        images = (images - images.mean()) / images.std()
        for k, v in maps.items():
            maps[k] = v.to(device="cuda", dtype=torch.float32)

        preds = model(images)
        loss = loss_fn(preds, maps, train=True)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    scheduler.step()
    return running_loss / dataset_size


def val_epoch(model, val_loader, loss_fn):
    return 10.0


def main():
    epochs = 500

    model = DBNet().cuda()
    # checkpoint = torch.load('checkpoints/dbnet_047', weights_only=False)
    # model.load_state_dict(checkpoint['model_state_dict'])

    train_loader, val_loader = get_loaders("../data", batch_size=16)
    loss_fn = db_loss

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.007, weight_decay=1e-6)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.007, weight_decay=0.0001, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.007)
    # lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, 75 * epochs, 0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)

    best_loss = float("inf")

    run_id = get_run_id()
    print(f'starting run {run_id}')
    writer = SummaryWriter(f"../runs/{run_id}")

    for epoch in range(epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn)
        val_loss = val_epoch(model, val_loader, loss_fn)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"checkpoints/{run_id}",
            )

        print(
            f"[Epoch {
                epoch + 1}] train-loss: {train_loss:.4f} | lr: {scheduler.get_last_lr()[0]:.5f}"
        )
        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
