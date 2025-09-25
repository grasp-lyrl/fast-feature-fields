import cv2
import yaml
import torch
import argparse
import datetime
import numpy as np

from f3.utils import num_params, setup_torch, setup_experiment
from f3.tasks.optical_flow.utils import (EventFFFlow, EventFlow,
                                             draw_color_wheel_np,
                                             get_dataloaders_from_args_evev,
                                             train_fixed_time_optical_flow as train,
                                             validate_fixed_time_optical_flow as validate)


parser = argparse.ArgumentParser("Train a Optical Flow model on a dataset of events.")

parser.add_argument("--wandb", action="store_true", help="Log to wandb.")
parser.add_argument("--name", type=str, help="Name of the run.")
parser.add_argument("--conf", type=str, required=True, help="Config file for Optical Flow + EventFF model,ckpt and dataset files")
parser.add_argument("--compile", action="store_true", help="Torch compile both the optical flow model")
parser.add_argument("--baseline", action="store_true", help="Train the baseline model.")
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
        args.name = f"f3flow_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base_path = f"outputs/optical_flow/{args.name}"
    models_path = base_path + "/models"

    logger, resume = setup_experiment(args, base_path, models_path)

    train_loader, val_loader = get_dataloaders_from_args_evev(args, logger)

    logger.info("#"*50)
    if not args.baseline:
        model = EventFFFlow(args.eventff["config"], pyramids=args.pyramids, alpha=args.alpha,
                            flowhead_config=args.flowhead, return_loss=True)
        model.eventff = torch.compile(model.eventff)
        model.load_weights(args.eventff["ckpt"])
    else:
        model = EventFlow(args.eventmodel, *args.frame_sizes, args.time_ctx // args.bucket,
                          pyramids=args.pyramids, alpha=args.alpha, flowhead_config=args.flowhead, return_loss=True)
        if args.compile:
            model.upchannel = torch.compile(model.upchannel)
    if args.compile:
        model.flowhead = torch.compile(model.flowhead)
    model.save_configs(models_path)

    logger.info(f"Feature Field + Optical Flow: {model}")
    logger.info(f"Trainable parameters in Flow Head: {num_params(model.flowhead)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info(f"Train datasets: {' '.join(args.train['datasets'])}")
    logger.info("#"*50)
    
    cv2.imwrite(f"{base_path}/color_wheel.png", draw_color_wheel_np(512, 512))

    scaler = torch.GradScaler(enabled=args.amp)

    param_groups = [
        {"params": model.flowhead.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.lr_end_factor, total_iters=args.epochs) # 3e-3 -> 3e-4 default

    start, best_loss, best_phto_loss, best_smooth_loss = 0, np.inf, np.inf, np.inf
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
            best_phto_loss = best_dict["photometric_loss"]
            best_smooth_loss = best_dict["smoothness_loss"]
            del best_dict
        except FileNotFoundError:
            pass

        torch.cuda.empty_cache()
        logger.info(f"Resuming from epoch: {start}, Best Loss: {best_loss}, Best Photometric Loss: {best_phto_loss}, Best Smoothness Loss: {best_smooth_loss}")

    val_loss, val_phto_loss, val_smooth_loss = 0, 0, 0

    for epoch in range(start, args.epochs):
        train(args, model, train_loader, optimizer, scheduler, epoch, logger=logger,
              scaler=scaler, iters_to_accumulate=args.train["batch"] // args.train["mini_batch"])
        if (epoch+1) % args.val_interval == 0:
            save_preds = True if (epoch+1) % args.log_interval == 0 else False
            with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
                val_loss, val_phto_loss, val_smooth_loss = validate(args, model, val_loader, epoch, logger=logger, save_preds=save_preds)
            if val_loss < best_loss:
                best_loss = val_loss
                best_dict = {
                    "epoch": epoch,
                    "loss": best_loss,
                    "photometric_loss": val_phto_loss,
                    "smoothness_loss": val_smooth_loss,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(best_dict, f"{models_path}/best.pth")
                logger.info(f"Saving best model at epoch: {epoch}, Loss: {best_loss}, " +\
                            f"Photometric Loss: {val_phto_loss}, Smoothness Loss: {val_smooth_loss}")
            if val_phto_loss < best_phto_loss:
                best_phto_loss = val_phto_loss
                best_dict = {
                    "epoch": epoch,
                    "loss": val_loss,
                    "photometric_loss": best_phto_loss,
                    "smoothness_loss": val_smooth_loss,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(best_dict, f"{models_path}/best_phto.pth")
                logger.info(f"Saving best photometric loss model at epoch: {epoch}, Loss: {val_loss}, " +\
                            f"Photometric Loss: {best_phto_loss}, Smoothness Loss: {val_smooth_loss}")

        scheduler.step()
        last_dict = {
            "epoch": epoch,
            "loss": val_loss,
            "photometric_loss": val_phto_loss,
            "smoothness_loss": val_smooth_loss,
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
