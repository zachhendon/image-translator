import os
import sys

sys.path.append(os.path.abspath(".."))

import torch
from torch.nn import BCELoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
from model import DBNet
from data.dataloader import get_loaders
import glob
from tqdm import tqdm

bin_loss_fn = BCELoss()
prob_loss_fn = BCELoss()
thresh_loss_fn = L1Loss()


def db_loss(pred, gt, train=True):
    if train:
        bin_loss = bin_loss_fn(pred["bin_map"], gt["bin_map"])
        prob_loss = prob_loss_fn(pred["prob_map"], gt["prob_map"])
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


def train_epoch(model, train_loader, optimizer, loss_fn):
    model.train()

    running_loss = 0
    dataset_size = 0

    for images, maps in tqdm(train_loader):
        optimizer.zero_grad()
        batch_size = len(images)

        images = images.to(device="cuda", dtype=torch.float32)
        images = images.permute(0, 3, 1, 2)
        for k, v in maps.items():
            maps[k] = v.to(device="cuda", dtype=torch.float32)

        preds = model(images)
        loss = loss_fn(preds, maps, train=True)
        loss.backward()

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size
    torch.cuda.empty_cache()
    return running_loss / dataset_size


def val_epoch(model, val_loader, loss_fn):
    return 10.0


def main():
    run_id = get_run_id()
    writer = SummaryWriter(f"../runs/{run_id}")
    epochs = 3

    model = DBNet().cuda()
    train_loader, val_loader = get_loaders("../data", batch_size=16)
    loss_fn = db_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
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
            f"[Epoch {epoch + 1}] train-loss: {train_loss:.4f} | val-loss: {val_loss:.4f}"
        )
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
