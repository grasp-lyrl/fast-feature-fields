import yaml
import torch
import argparse
import datetime
import numpy as np

from f3 import init_event_model, load_weights_ckpt
from f3.utils import (get_dataloaders_from_args, setup_torch, setup_accelerate_experiment,
                          num_params, train_fixed_time as train, validate_fixed_time as validate)


parser = argparse.ArgumentParser("Train a feature field on a dataset of events.")

parser.add_argument("--wandb", action="store_true", help="Log to wandb.")
parser.add_argument("--name", type=str, help="Name of the run.")
parser.add_argument("--conf", type=str, required=True, help="Path to the config file. If provided will load configs from here."+\
                                                            "Rest of the configs will be loaded from the defaults below")
parser.add_argument("--init", type=str, default=None, help="Path to the model weights to initialize from.")
parser.add_argument("--compile", action="store_true", help="Compile the model for faster training.")

args = parser.parse_args()
keys = set(vars(args).keys())

with open(args.conf, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


def main():
    setup_torch(cudnn_benchmark=False)

    if args.name is None:
        args.name = f"f3_{datetime.datetime.now().strftime('%Y%m%d-%H%M')}"
    base_path = f"outputs/{args.name}"
    models_path = f"outputs/{args.name}/models"
    
    logger, resume, accelerator, device, gradient_accumulation_steps = setup_accelerate_experiment(args, base_path, models_path)

    train_loader, val_loader = get_dataloaders_from_args(args, logger)
    logger.info(f"Train datasets: {' '.join(args.train['datasets'])}")

    logger.info("#"*50)
    eff = init_event_model(args.eventff["config"], return_logits=True, return_loss=True, loss_fn=args.loss_fn).to(device)
    eff.save_config(f"{models_path}/config.yml")
    eff_uncompiled = eff  # Keep reference to non-compiled model
    if args.compile: eff = torch.compile(eff, fullgraph=False)

    logger.info(f"Feature Field: {eff}")
    if hasattr(eff, "multi_hash_encoder"):
        logger.info(f"Trainable parameters in the Multi Resolution Hash Encoder: {num_params(eff.multi_hash_encoder)}")
    logger.info(f"Total Trainable parameters: {num_params(eff)}")
    logger.info("#"*50)

    optimizer = torch.optim.AdamW(eff.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.lr_end_factor, total_iters=args.epochs) # 3e-3 -> 3e-4 default

    start, best_loss, best_acc, best_f1 = 0, np.inf, 0, 0
    if resume:
        last_dict = torch.load(f"{models_path}/last.pth", weights_only=True)
        eff.load_state_dict(last_dict["model"])
        optimizer.load_state_dict(last_dict["optimizer"])
        scheduler.load_state_dict(last_dict["scheduler"])
        start = last_dict["epoch"] + 1
        del last_dict

        try:
            best_dict = torch.load(f"{models_path}/best.pth", weights_only=True)
            best_loss = best_dict["loss"]
            best_acc = best_dict["acc"]
            best_f1 = best_dict["f1"]
            del best_dict
        except FileNotFoundError:
            best_loss, best_acc, best_f1 = np.inf, 0, 0

        torch.cuda.empty_cache()
        logger.info(f"Resuming from epoch: {start}, Best Loss: {best_loss}, Best Acc: {best_acc}, Best F1: {best_f1}")
    else:
        if args.init is not None:
            epoch, loss, acc = load_weights_ckpt(eff, args.init, strict=True)
            logger.info(f"Loaded weights from {args.init} at epoch: {epoch}, Loss: {loss}, Acc: {acc}")
        else:
            logger.info("No weights to load, starting from scratch.")


    eff_uncompiled, eff, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        eff_uncompiled, eff, optimizer, train_loader, val_loader, scheduler
    )

    if resume: val_loss, val_acc, val_f1 = np.inf, 0, 0
    else:      val_loss, val_acc, val_f1 = validate(args, eff, val_loader, start, accelerator=accelerator, logger=logger)

    for epoch in range(start, args.epochs):
        train(args, eff, train_loader, optimizer, scheduler, epoch, logger=logger,
              accelerator=accelerator, iters_to_accumulate=gradient_accumulation_steps)

        accelerator.wait_for_everyone()
        unwrapped_eff = accelerator.unwrap_model(eff)
        if args.compile:
            unwrapped_eff_uncompiled = accelerator.unwrap_model(eff_uncompiled)

        if (epoch+1) % args.val_interval == 0:
            save_preds = True if (epoch+1) % args.log_interval == 0 else False
            val_loss, val_acc, val_f1 = validate(args, eff, val_loader, epoch,
                                                 accelerator=accelerator, logger=logger, save_preds=save_preds)

            accelerator.wait_for_everyone()
            unwrapped_eff = accelerator.unwrap_model(eff)
            if args.compile:
                unwrapped_eff_uncompiled = accelerator.unwrap_model(eff_uncompiled)

            if val_loss < best_loss:
                best_loss = val_loss
                best_acc = val_acc
                accelerator.save({
                    "epoch": epoch,
                    "loss": best_loss,
                    "acc": best_acc,
                    "f1": val_f1,
                    "model": unwrapped_eff.state_dict(),
                    "optimizer": optimizer.optimizer.state_dict(),
                    "scheduler": scheduler.scheduler.state_dict()
                }, f"{models_path}/best.pth")
                if args.compile:
                    accelerator.save(unwrapped_eff_uncompiled.state_dict(), f"{models_path}/best_uncompiled.pth")
                if accelerator.is_local_main_process:
                    logger.info(f"Saving best model at epoch: {epoch}, Loss: {best_loss}, Acc: {best_acc}")
            if val_f1 > best_f1:
                best_f1 = val_f1
                accelerator.save({
                    "epoch": epoch,
                    "loss": val_loss,
                    "acc": val_acc,
                    "f1": best_f1,
                    "model": unwrapped_eff.state_dict(),
                    "optimizer": optimizer.optimizer.state_dict(),
                    "scheduler": scheduler.scheduler.state_dict()
                }, f"{models_path}/best_f1.pth")
                if args.compile:
                    accelerator.save(unwrapped_eff_uncompiled.state_dict(), f"{models_path}/best_f1_uncompiled.pth")
                if accelerator.is_local_main_process:
                    logger.info(f"Saving best f1 model at epoch: {epoch}, Loss: {best_loss}, Acc: {best_acc}, F1: {best_f1}")

        scheduler.step()
        accelerator.save({
            "epoch": epoch,
            "loss": val_loss,
            "acc": val_acc,
            "f1": val_f1,
            "model": unwrapped_eff.state_dict(),
            "optimizer": optimizer.optimizer.state_dict(),
            "scheduler": scheduler.scheduler.state_dict()
        }, f"{models_path}/last.pth")
        if args.compile:
            accelerator.save(unwrapped_eff_uncompiled.state_dict(), f"{models_path}/last_uncompiled.pth")

        # For long runs we want intermediate checkpoints
        if (epoch+1) % args.log_interval == 0:
            accelerator.save({
                "epoch": epoch,
                "loss": val_loss,
                "acc": val_acc,
                "f1": val_f1,
                "model": unwrapped_eff.state_dict(),
                "optimizer": optimizer.optimizer.state_dict(),
                "scheduler": scheduler.scheduler.state_dict()
            }, f"{models_path}/checkpoint_{epoch}.pth")
            if args.compile:
                accelerator.save(unwrapped_eff_uncompiled.state_dict(), f"{models_path}/checkpoint_{epoch}_uncompiled.pth")
    accelerator.end_training()


if __name__ == '__main__':
    main()
