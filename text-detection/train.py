import torch
import argparse
import os
import os.path as osp
import wandb
import sys
import ast
from models.utils import evaluate_micro, get_iou
from hydra import compose, initialize
from models.fast import FAST
from models.loss.ohem import get_ohem_masks
from models.loss.bce_loss import BCELoss
from models.loss.dice_loss import DiceLoss
from models.loss.auf_loss import AUFLoss
from models.loss.weighted_cross_entropy import WeightedCrossEntropyLoss
from models.loss.ce_border import CEBorderLoss
from dali_loader import get_loader


def main(args):
    # load config and download model artifacts from wandb
    if args.config:
        with initialize(config_path="config", version_base=None):
            cfg = dict(compose(config_name=args.config))

        run = wandb.init(
            project="text-detection", config=cfg, settings=wandb.Settings(code_dir=".")
        )
        if args.checkpoint:
            artifact = run.use_artifact(
                f"{args.checkpoint}-model-best.pt:latest", type="model"
            )
            artifact_path = artifact.download()
        else:
            wandb.run.log_code(".")
    else:
        run = wandb.init(project="text-detection", resume="must", id=args.resume)
        artifact = run.use_artifact(f"{run.id}-model-recent.pt:latest", type="model")
        artifact_path = artifact.download()
        cfg = {k: ast.literal_eval(v) for k, v in run.config.items()}

    # data loaders
    batch_size = cfg["data"]["batch_size"]
    train_loader = get_loader(
        cfg["data"]["dataset"], "train", 640, batch_size=batch_size, num_threads=1
    )
    val_loader = get_loader(
        cfg["data"]["dataset"], "val", 736, batch_size=1, num_threads=1
    )

    # training parameters
    num_iterations = cfg["train"]["num_iterations"]
    if cfg["train"]["train_interval"]:
        train_interval = cfg["train"]["train_interval"]
    else:
        train_interval = len(train_loader)
    if cfg["train"]["val_interval"]:
        val_interval = cfg["train"]["val_interval"]
    else:
        val_interval = len(val_loader)
    val_interval *= batch_size  # account for batch size of 1 in validation loader
    eval_interval = cfg["eval"]["eval_interval"]
    if eval_interval % train_interval != 0:
        print("Error: eval_interval must be multiple of train_interval")
        sys.exit(1)

    # model
    if cfg["model"]["type"] == "fast":
        model = FAST()
    model = model.cuda()

    # # kernel loss
    # kernel_loss_str = cfg["model"]["kernel_loss"]["loss_fn"]
    # if kernel_loss_str == "bce_loss":
    #     kernel_loss_fn = BCELoss()
    # elif kernel_loss_str == "dice_loss":
    #     kernel_loss_fn = DiceLoss()
    # elif kernel_loss_str == "auf_loss":
    #     kernel_loss_fn = AUFLoss(gamma=cfg["model"]["kernel_loss"]["gamma"])
    # elif kernel_loss_str == "cross_entropy":
    #     kernel_loss_fn = WeightedCrossEntropyLoss()
    # else:
    #     print(f"Error: kernel loss function {kernel_loss_str} not recognized")
    #     sys.exit(1)
    # kernel_weight = cfg["model"]["kernel_loss"]["weight"] or 1
    # kernel_ohem = cfg["model"]["kernel_loss"]["ohem"]

    # # text loss
    # text_loss_str = cfg["model"]["text_loss"]["loss_fn"]
    # if text_loss_str == "bce_loss":
    #     text_loss_fn = BCELoss()
    # elif text_loss_str == "dice_loss":
    #     text_loss_fn = DiceLoss()
    # elif text_loss_str == "auf_loss":
    #     text_loss_fn = AUFLoss(gamma=cfg["model"]["text_loss"]["gamma"])
    # elif text_loss_str == "cross_entropy":
    #     text_loss_fn = WeightedCrossEntropyLoss()
    # else:
    #     print(f"Error: text loss function {text_loss_str} not recognized")
    #     sys.exit(1)
    # text_weight = cfg["model"]["text_loss"]["weight"] or 1
    # text_ohem = cfg["model"]["text_loss"]["ohem"]
    
    if cfg["model"]["loss_fn"] == "ce_border":
        loss_fn = CEBorderLoss()
    else:
        print(f"Error: loss function {cfg['model']['loss_fn']} not recognized")
        sys.exit(1)

    # optimizer
    if cfg["train"]["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    elif cfg["train"]["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg["train"]["lr"],
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
            fused=True,
        )

    # learning rate scheduler
    if not cfg["train"]["warmup_iterations"]:
        warmup_iterations = 0
    else:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            total_iters=cfg["train"]["warmup_iterations"],
        )
        warmup_iterations = cfg["train"]["warmup_iterations"]

    if cfg["train"]["schedule"] == "poly":
        scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, power=0.9, total_iters=num_iterations - warmup_iterations
        )
    if cfg["train"]["warmup_iterations"]:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, scheduler], [warmup_iterations]
        )

    # load model weights if resuming or starting from checkpoint
    iteration = 0
    best_f1 = 0.0
    if args.resume:
        checkpoint = torch.load(
            osp.join(artifact_path, "recent.pth"), weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        iteration = checkpoint["iteration"]
        best_f1 = checkpoint["best_f1"]
        print(f"Resuming run {run.id} at iteration {iteration}")
    elif args.checkpoint:
        checkpoint = torch.load(osp.join(artifact_path, "best.pth"), weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_f1 = checkpoint["best_f1"]
        print(f"Loaded checkpoint {run.id}")
    checkpoint_path = osp.join("checkpoints", run.id)
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Local model directory: {checkpoint_path}")

    # get binarization threshold for evaluation
    threshold = cfg["eval"]["threshold"]

    # train for num_iterations or until interrupted
    while iteration < num_iterations:
        # train loop
        model.train()
        torch.cuda.empty_cache()
        # train_kernel_running_loss = 0.0
        # train_text_running_loss = 0.0
        train_running_loss = 0.0

        for _ in range(train_interval):
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

            # if kernel_ohem:
            #     ohem_kernel_mask = get_ohem_masks(preds, gt_kernels, training_masks)
            #     loss_kernel = kernel_loss_fn(preds, gt_kernels, ohem_kernel_mask)
            # else:
            #     loss_kernel = kernel_loss_fn(preds, gt_kernels, training_masks)
            # if text_ohem:
            #     ohem_text_mask = get_ohem_masks(dilated_preds, gt_texts, training_masks)
            #     loss_text = text_loss_fn(dilated_preds, gt_texts, ohem_text_mask)
            # else:
            #     loss_text = text_loss_fn(dilated_preds, gt_texts, training_masks)
            # loss = calculate_total_loss(loss_kernel, loss_text)
            # loss = loss_kernel * kernel_weight + loss_text * text_weight
            loss = loss_fn(preds, gt_kernels, gt_texts, training_masks)
            loss.backward()

            optimizer.step()
            scheduler.step()

            # train_kernel_running_loss += loss_kernel.item()
            # train_text_running_loss += loss_text.item()
            train_running_loss += loss.item()
        # train_kernel_loss = train_kernel_running_loss / train_interval
        # train_text_loss = train_text_running_loss / train_interval
        train_loss = train_running_loss / train_interval

        # validation loop
        model.eval()
        torch.cuda.empty_cache()
        # val_kernel_running_loss = 0.0
        # val_text_running_loss = 0.0
        val_running_loss = 0.0
        val_running_kernel_iou = 0.0
        val_running_text_iou = 0.0

        with torch.no_grad():
            for _ in range(val_interval):
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

                # if kernel_ohem:
                #     ohem_kernel_mask = get_ohem_masks(preds, gt_kernels, training_masks)
                #     loss_kernel = kernel_loss_fn(preds, gt_kernels, ohem_kernel_mask)
                # else:
                #     loss_kernel = kernel_loss_fn(preds, gt_kernels, training_masks)
                # if text_ohem:
                #     ohem_text_mask = get_ohem_masks(
                #         dilated_preds, gt_texts, training_masks
                #     )
                #     loss_text = text_loss_fn(dilated_preds, gt_texts, ohem_text_mask)
                # else:
                #     loss_text = text_loss_fn(dilated_preds, gt_texts, training_masks)
                # # loss = calculate_total_loss(loss_kernel, loss_text)
                # loss = loss_kernel * kernel_weight + loss_text * text_weight
                loss = loss_fn(preds, gt_kernels, gt_texts, training_masks)

                # val_kernel_running_loss += loss_kernel.item()
                # val_text_running_loss += loss_text.item()
                val_running_loss += loss.item()

                val_running_kernel_iou += get_iou(preds[:, 0], gt_kernels, training_masks)
                val_running_text_iou += get_iou(dilated_preds[:, 0], gt_texts, training_masks)

            # val_kernel_loss = val_kernel_running_loss / val_interval
            # val_text_loss = val_text_running_loss / val_interval
            val_loss = val_running_loss / val_interval
            val_kernel_iou = val_running_kernel_iou / val_interval
            val_text_iou = val_running_text_iou / val_interval

        iteration += train_interval

        # create wandb log and print
        log = {
            "iteration": iteration,
            "train_loss": train_loss,
            "val_loss": val_loss,
            # "kernel_loss_train": train_kernel_loss,
            # "kernel_loss_val": val_kernel_loss,
            # "text_loss_train": train_text_loss,
            # "text_loss_val": val_text_loss,
            "kernel_iou": val_kernel_iou,
            "text_iou": val_text_iou,
        }
        print(
            f"[Iter {iteration}] | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | IOU(kernel/text): {val_kernel_iou:.4f}/{val_text_iou:.4f} | lr: {scheduler.get_last_lr()[0]:.7f}"
        )

        # evaluate model
        if iteration % eval_interval == 0:
            precision, recall, f1 = evaluate_micro(
                model, val_loader, threshold=threshold, iter_limit=val_interval
            )
            log["precision"] = precision
            log["recall"] = recall
            log["f1"] = f1
            print(f"precision: {precision:.4f} | recall: {recall:.4f} | f1: {f1:.4f}")
        # log metrics to wandb
        wandb.log(log)

        # create and save recent/best checkpoints
        if iteration % eval_interval == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "iteration": iteration,
                "best_f1": f1,
            }

            # save most recent checkpoint
            save_path_recent = osp.join(checkpoint_path, "recent.pth")
            torch.save(checkpoint, save_path_recent)
            artifact = wandb.Artifact(
                f"{run.id}-model-recent.pt", type="model", metadata=log
            )
            artifact.add_file(save_path_recent)
            wandb.log_artifact(artifact)

            # save best checkpoint if f1 improved
            if f1 > best_f1:
                best_f1 = f1
                save_path_best = osp.join(checkpoint_path, "best.pth")
                torch.save(checkpoint, save_path_best)
                artifact = wandb.Artifact(
                    f"{run.id}-model-best.pt", type="model", metadata=log
                )
                artifact.add_file(save_path_best)
                wandb.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument("-c", "--config", type=str, help="config file path")
    parser.add_argument(
        "-r", "--resume", type=str, help="wandb resume id to resume training from"
    )
    parser.add_argument(
        "-s",
        "--checkpoint",
        type=str,
        help="wandb checkpoint id to start training from",
    )
    args = parser.parse_args()

    if not args.config and not args.resume:
        print("Error: need to specify config or resume id")
        sys.exit(1)
    elif args.config and args.resume:
        print("Error: cannot specify both config and resume id")
        sys.exit(1)
    elif args.checkpoint and not args.config:
        print("Error: need to specify config when starting from checkpoint")
        sys.exit(1)
    elif args.checkpoint and args.resume:
        print("Error: cannot specify both checkpoint and resume id")
        sys.exit(1)

    main(args)
