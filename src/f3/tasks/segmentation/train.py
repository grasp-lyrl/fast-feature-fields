import yaml
import argparse
import datetime
import numpy as np

import torch
import torch._dynamo
import torch.nn as nn

from f3.utils import num_params, setup_torch, setup_experiment
from f3.tasks.segmentation.utils import (EventFFSegformer,
                                             get_dataloaders_from_args,
                                             train_fixed_time_segmentation as train,
                                             validate_fixed_time_segmentation as validate)


parser = argparse.ArgumentParser("Train a segmentation model on a dataset of events.")

parser.add_argument("--wandb", action="store_true", help="Log to wandb.")
parser.add_argument("--name", type=str, help="Name of the run.")
parser.add_argument("--conf", type=str, required=True, help="Config file for Segformer + EventFF model,ckpt and dataset files")
parser.add_argument("--compile", action="store_true", help="Torch compile both the segmentation model")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")

args = parser.parse_args()
keys = set(vars(args).keys())

with open(args.conf, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


def main():
    setup_torch(cudnn_benchmark=True)

    if args.name is None:
        args.name = f"f3seg_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base_path = f"outputs/segmentation/{args.name}"
    models_path = f"outputs/segmentation/{args.name}/models"

    logger, resume = setup_experiment(args, base_path=base_path, models_path=models_path)

    train_loader, val_loader = get_dataloaders_from_args(args, logger)

    logger.info("#"*50)
    model = EventFFSegformer(args.eventff["config"], args.segformer_config, args.num_labels)
    if not args.compile:
        model.eventff = torch.compile(model.eventff)
    else:
        model = torch.compile(model)
    model.load_weights(args.eventff["ckpt"])
    model.save_configs(models_path)

    logger.info(f"Feature Field + Segmentation: {model}")
    logger.info(f"Trainable parameters in Segformer: {num_params(model.segformer)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info(f"Train datasets: {' '.join(args.train['datasets'])}")
    logger.info("#"*50)

    scaler = torch.GradScaler(enabled=args.amp)
    
    #! Dont weight decay the norm layers and have higher lr for decoder head
    param_groups = []
    for name, param in model.named_parameters():
        if 'pos_block' in name:
            param_groups.append({'params': param, 'weight_decay': 0.0})
        elif 'norm' in name:
            param_groups.append({'params': param, 'weight_decay': 0.0})
        elif 'head' in name:
            param_groups.append({'params': param, 'lr': 5*args.lr})
        elif 'eventff' in name:
            param_groups.append({'params': param, 'lr': 0.0})
        else:
            param_groups.append({'params': param})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.lr_end_factor, total_iters=args.epochs) # 3e-3 -> 3e-4 default

    start, best_loss, best_acc, best_miou = 0, np.inf, 0, 0
    if resume:
        last_dict = torch.load(f"{models_path}/last.pth")
        model.load_state_dict(last_dict["model"])
        optimizer.load_state_dict(last_dict["optimizer"])
        scheduler.load_state_dict(last_dict["scheduler"])
        start = last_dict["epoch"] + 1
        del last_dict

        try:
            best_dict = torch.load(f"{models_path}/best.pth")
            best_loss = best_dict["loss"]
            best_acc = best_dict["acc"]
            best_miou = best_dict["miou"]
            del best_dict
        except FileNotFoundError:
            best_loss, best_acc, best_miou = np.inf, 0, 0

        torch.cuda.empty_cache()
        logger.info(f"Resuming from epoch: {start}, Best Loss: {best_loss}, Best Acc: {best_acc}, Best MIoU: {best_miou}")

    val_loss, val_acc, val_miou = np.inf, 0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    
    for epoch in range(start, args.epochs):
        train(args, model, train_loader, optimizer, scheduler, loss_fn, epoch, logger=logger,
              scaler=scaler, iters_to_accumulate=args.train["batch"] // args.train["mini_batch"])
        if (epoch+1) % args.val_interval == 0:
            save_preds = True if (epoch+1) % args.log_interval == 0 else False
            with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
                val_loss, val_acc, val_miou = validate(args, model, val_loader, loss_fn, epoch, logger=logger, save_preds=save_preds)
            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                best_dict = {
                    "epoch": epoch,
                    "loss": best_loss,
                    "acc": best_acc,
                    "miou": val_miou,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(best_dict, f"{models_path}/best.pth")
                logger.info(f"Saving best model at epoch: {epoch}, Loss: {best_loss}, Acc: {best_acc}")
            if val_miou > best_miou:
                best_miou = val_miou
                best_miou_dict = {
                    "epoch": epoch,
                    "loss": val_loss,
                    "acc": val_acc,
                    "miou": best_miou,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(best_miou_dict, f"{models_path}/best_miou.pth")
                logger.info(f"Saving best MIoU model at epoch: {epoch}, Loss: {best_loss}, Acc: {best_acc}, MIoU: {best_miou}")

        scheduler.step()
        last_dict = {
            "epoch": epoch,
            "loss": val_loss,
            "acc": val_acc,
            "miou": val_miou,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(last_dict, f"{models_path}/last.pth")

        # For long runs we want intermediate checkpoints
        if (epoch+1) % args.log_interval == 0:
            torch.save(last_dict, f"{models_path}/checkpoint_{epoch}.pth")


if __name__ == '__main__':
    main()
