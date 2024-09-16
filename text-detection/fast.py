import torch
import torch.nn as nn
import time
from data.dataloader import get_loaders
from torchvision.ops import FeaturePyramidNetwork
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import glob


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fpn = FeaturePyramidNetwork([64, 128, 256, 512], 64)

    def forward(self, x):
        features = {}
        x = self.conv1(x)
        x = self.stage1(x)
        features["stage1"] = x
        x = self.stage2(x)
        features["stage2"] = x
        x = self.stage3(x)
        features["stage3"] = x
        x = self.stage4(x)
        features["stage4"] = x

        return features


class Neck(nn.Module):
    def __init__(self):
        super().__init__()

        self.reduce2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.reduce3 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.reduce4 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.reduce5 = nn.Conv2d(512, 128, 3, padding=1, bias=False)

    def forward(self, x):
        c2, c3, c4, c5 = x.values()

        p2 = self.reduce2(c2)
        p3 = self.reduce3(c3)
        p3 = F.interpolate(p3, scale_factor=2, mode="bilinear")
        p4 = self.reduce4(c4)
        p4 = F.interpolate(p4, scale_factor=4, mode="bilinear")
        p5 = self.reduce5(c5)
        p5 = F.interpolate(p5, scale_factor=8, mode="bilinear")

        p = torch.cat([p2, p3, p4, p5], 1)
        return p


loss_kernel_fn = nn.BCELoss()
loss_text_fn = nn.BCELoss()


class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(512, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 2, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.conv(x)

    def get_unified_focal_loss(self, pred, true):
        delta = 0.6
        gamma = 0.5
        lmd = 0.5

        pt = true * pred + (1 - true) * (1 - pred)
        L_maF = -delta * true * torch.log(pt + 1e-8) - (1 - delta) * torch.pow(
            pred + 1e-8, gamma
        ) * torch.log(pt + 1e-8)
        L_maF = -delta * pred * torch.log(pt + 1e-8)
        L_maF = L_maF.mean()

        mTI = (pred * true).sum() / (
            pred * true
            + delta * (pred * (1 - true))
            + (1 - delta) * ((1 - pred) * true)
        ).sum()
        L_maFT = (1 - mTI) + torch.pow(1 - mTI + 1e-8, 1 - gamma)

        return lmd * L_maF + (1 - lmd) * L_maFT
    
    def get_unified_focal_loss_sym(self, pred, true):
        delta = 0.6
        gamma = 0.5
        lmd = 0.5
        
        pt = true * pred + (1 - true) * (1 - pred)
        L_mF = delta * (1 - pt + 1e-8).pow(gamma) * F.binary_cross_entropy(pred, true, reduction='none')
        L_mF = L_mF.mean()
        
        mTI = (pred * true).sum() / (
            pred * true
            + delta * (pred * (1 - true))
            + (1 - delta) * ((1 - pred) * true)
        ).sum()
        L_mFT = (1 - mTI).pow(gamma)
        
        return lmd * L_mF + (1 - lmd) * L_mFT

    def loss(self, out, gt_kernels, gt_texts):
        loss_kernel_old = loss_kernel_fn(out, gt_kernels)
        # loss_kernel = 1 - (2 * (out * gt_kernels).sum()) / (torch.pow(out, 2).sum() + torch.pow(gt_kernels, 2).sum())
        loss_kernel = self.get_unified_focal_loss_sym(out, gt_kernels)

        pred_text = F.max_pool2d(out, 9, stride=1, padding=4)
        loss_text_old = loss_text = loss_text_fn(pred_text, gt_texts)
        # loss_text = 1 - (2 * (pred_text * gt_texts).sum()) / (torch.pow(pred_text, 2).sum() + torch.pow(gt_texts, 2).sum())
        loss_text = self.get_unified_focal_loss_sym(pred_text, gt_texts)

        return loss_kernel + 0.5 * loss_text, loss_kernel_old + 0.5 * loss_text_old


def timing_decorator(func):
    def wrapper(self, *args, **kwargs):
        if self.training:
            return func(self, *args, **kwargs)
        else:
            torch.cuda.synchronize()
            start = time.time()
            x = func(self, *args, **kwargs)
            torch.cuda.synchronize()
            end = time.time()
            return x, end - start

    return wrapper


class FAST(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head()

    @timing_decorator
    def run_backbone(self, images):
        return self.backbone(images)

    @timing_decorator
    def run_neck(self, x):
        return self.neck(x)

    @timing_decorator
    def run_head(self, x):
        return self.head(x)

    def forward(self, images, gt_kernels, gt_texts):
        outputs = {}

        if self.training:
            x = self.run_backbone(images)
            x = self.run_neck(x)
            x = self.run_head(x)

            outputs["loss"] = self.head.loss(x, gt_kernels, gt_texts)
        else:
            x, outputs["backbone_time"] = self.run_backbone(images)
            x, outputs["neck_time"] = self.run_neck(x)
            x, outputs["head_time"] = self.run_head(x)
            outputs["output"] = x
            outputs["loss"] = self.head.loss(x, gt_kernels, gt_texts)
        return outputs


def train_epoch(model, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    running_loss_old = 0.0
    dataset_size = 0

    for images, maps in train_loader:
        images = images.to(dtype=torch.float32, device="cuda")
        eroded_maps = maps["eroded_maps"].to(dtype=torch.float32, device="cuda")
        gt_maps = maps["gt_maps"].to(dtype=torch.float32, device="cuda")

        optimizer.zero_grad()
        batch_size = len(images)

        output = model(images, eroded_maps, gt_maps)
        loss, loss_old = output["loss"]
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        running_loss_old += loss_old.item() * batch_size
        dataset_size += batch_size
    return running_loss / dataset_size, running_loss_old / dataset_size


def val_epoch(model, val_loader):
    model.eval()
    running_loss = 0.0
    running_loss_old = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, maps in val_loader:
            images = images.to(dtype=torch.float32, device="cuda")
            eroded_maps = maps["eroded_maps"].to(dtype=torch.float32, device="cuda")
            gt_maps = maps["gt_maps"].to(dtype=torch.float32, device="cuda")

            batch_size = len(images)

            output = model(images, eroded_maps, gt_maps)
            loss, loss_old = output["loss"]

            running_loss += loss.item() * batch_size
            running_loss_old += loss_old.item() * batch_size
            dataset_size += batch_size
    return running_loss / dataset_size, running_loss_old / dataset_size


def get_run_id():
    prev_runs = glob.glob("runs/*")
    if len(prev_runs) == 0:
        prev_run_id = -1
    else:
        prev_run_ids = [int(run[-3:]) for run in prev_runs]
        prev_run_id = sorted(prev_run_ids)[-1]
    return f"fast_{str(prev_run_id + 1).zfill(3)}"


def main():
    epochs = 250

    train_loader, val_loader = get_loaders("data", batch_size=16, train=True)
    model = FAST().cuda()
    optimizer = torch.optim.AdamW(model.parameters())

    run_id = get_run_id()
    print(f"starting run {run_id}")
    writer = SummaryWriter(f"runs/{run_id}")

    # num_checkpoints = 0
    best_loss = float("inf")
    for epoch in range(epochs):
        train_loss, train_loss_old = train_epoch(model, train_loader, optimizer)
        val_loss, val_loss_old = val_epoch(model, val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                f"models/checkpoints/{run_id}",
            )
            # num_checkpoints += 1

        print(
            f"[Epoch {epoch + 1}] | train loss: {train_loss:.4f} | train loss old: {train_loss_old:.4f} | val loss: {val_loss:.4f} | val loss old: {val_loss_old:.4f}"
        )
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch + 1)
        writer.add_scalars(
            "loss_old", {"train": train_loss_old, "val": val_loss_old}, epoch + 1
        )
        writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
