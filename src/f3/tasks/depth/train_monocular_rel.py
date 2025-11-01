import copy
import yaml
import torch
import argparse
import datetime

from f3.utils import num_params, setup_torch, setup_experiment, log_dict
from f3.tasks.depth.utils import (EventFFDepthAnythingV2, EventDepthAnythingV2,
                                      ScaleAndShiftInvariantLoss,
                                      set_best_results, get_dataloaders_from_args,
                                      train_fixed_time_disparity as train,
                                      validate_fixed_time_disparity as validate)

parser = argparse.ArgumentParser("Train a Monocular Depth model on a dataset of events.")

parser.add_argument("--wandb", action="store_true", help="Log to wandb.")
parser.add_argument("--name", type=str, help="Name of the run.")
parser.add_argument("--conf", type=str, required=True, help="Config file for Monocular Depth + EventFF model,ckpt and dataset files")
parser.add_argument("--compile", action="store_true", help="Torch compile both the Monocular Depth model")
parser.add_argument("--baseline", action="store_true", help="Use the baseline model.")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")
parser.add_argument("--init", type=str, default=None, help="Path to the initial weights for the model.")


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
        args.name = f"f3depth_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base_path = f"outputs/monoculardepth/{args.name}"
    models_path = f"outputs/monoculardepth/{args.name}/models"
    
    logger, resume = setup_experiment(args, base_path, models_path)

    train_loader, val_loader = get_dataloaders_from_args(args, logger)

    logger.info("#"*50)
    if args.baseline:
        model = EventDepthAnythingV2(args.dav2_config, args.eventmodel,
                                     *args.frame_sizes, args.time_ctx // args.bucket)
        model_uncompiled = model  # Keep reference to non-compiled model
        if args.compile:
            model.dav2 = torch.compile(model, fullgraph=False)
    else:
        model = EventFFDepthAnythingV2(args.eventff["config"], args.dav2_config)
        model_uncompiled = model  # Keep reference to non-compiled model
        if not args.compile:
            model.eventff = torch.compile(model.eventff, fullgraph=False)
        else:
            model.eventff = torch.compile(model.eventff, fullgraph=False)
            model.dav2 = torch.compile(model.dav2)
        model.load_weights(args.eventff["ckpt"])
    model.save_configs(models_path)

    if args.init is not None:
        logger.info(f"Loading initial weights from {args.init}")
        state_dict = torch.load(args.init)
        model.load_state_dict(state_dict["model"])
        torch.cuda.empty_cache()
        logger.info("Initial weights loaded successfully.")
    else:
        logger.info("No initial weights provided, starting from scratch.")

    logger.info(f"Feature Field + Monocular Depth: {model}")
    logger.info(f"Trainable parameters in Depth Anything V2: {num_params(model.dav2)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info(f"Train datasets: {' '.join(args.train['datasets'])}")
    logger.info("#"*50)

    scaler = torch.GradScaler(enabled=args.amp)
    
    #! Dont weight decay the norm layers and have higher lr for decoder head
    param_groups = []
    for name, param in model.named_parameters():
        if 'pretrained' in name:
            # Since we initialize the first layer with non-ideal weights, we are okay with forgetting them
            if 'patch_embed.proj' in name:
                param_groups.append({'params': param, 'lr': args.lr, 'weight_decay': 0.0})
            else:
                param_groups.append({'params': param, 'lr': args.lr})
        else:
            param_groups.append({'params': param, 'lr': 10 * args.lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.lr_end_factor, total_iters=args.epochs)

    start = 0
    loss_fn = ScaleAndShiftInvariantLoss(alpha=args.alpha, scales=args.scales)
    best_results = {
        '1pe': 100.0, '2pe': 100.0, '3pe': 100.0, 'rmse': 100.0,
        'rmse_log': 100.0, 'log10': 100.0, 'silog': 100.0, loss_fn.name: 100.0
    }
    if resume:
        last_dict = torch.load(f"{models_path}/last.pth")
        model.load_state_dict(last_dict["model"])
        optimizer.load_state_dict(last_dict["optimizer"])
        scheduler.load_state_dict(last_dict["scheduler"])
        start = last_dict["epoch"] + 1
        del last_dict

        try:
            best_dict = torch.load(f"{models_path}/best.pth")
            best_results = best_dict["results"]
            del best_dict
        except FileNotFoundError:
            logger.info("No best model found, so falling back on default best results")

        torch.cuda.empty_cache()
        logger.info(f"Resuming from epoch: {start}")
        log_dict(logger, best_results)

    val_results = copy.deepcopy(best_results)

    for epoch in range(start, args.epochs):
        train(args, model, train_loader, optimizer, scheduler, loss_fn, epoch, logger=logger,
              scaler=scaler, iters_to_accumulate=args.train["batch"] // args.train["mini_batch"])
        if (epoch+1) % args.val_interval == 0:
            save_preds = True if (epoch+1) % args.log_interval == 0 else False
            with torch.autocast(device_type="cuda", enabled=args.amp, dtype=torch.float16):
                val_results = validate(args, model, val_loader, loss_fn, epoch, logger=logger, save_preds=save_preds)
            better_ssimae = val_results[loss_fn.name] < best_results[loss_fn.name]
            better_2pe = val_results["2pe"] < best_results["2pe"]
            set_best_results(best_results, val_results)
            if better_ssimae:
                best_dict = {
                    "epoch": epoch,
                    "results": best_results,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(best_dict, f"{models_path}/best.pth")
                if args.compile:
                    torch.save(model_uncompiled.state_dict(), f"{models_path}/best_uncompiled.pth")
                logger.info(f"Saving best model at epoch: {epoch},")
                log_dict(logger, best_results)
            if better_2pe:
                best_2pe_dict = {
                    "epoch": epoch,
                    "results": best_results,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()
                }
                torch.save(best_2pe_dict, f"{models_path}/best_2pe.pth")
                if args.compile:
                    torch.save(model_uncompiled.state_dict(), f"{models_path}/best_2pe_uncompiled.pth")
                logger.info(f"Saving best 2pe model at epoch: {epoch},")
                log_dict(logger, best_results)
                
        scheduler.step()
        last_dict = {
            "epoch": epoch,
            "results": val_results,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict()
        }
        torch.save(last_dict, f"{models_path}/last.pth")
        if args.compile:
            torch.save(model_uncompiled.state_dict(), f"{models_path}/last_uncompiled.pth")

        # For long runs we want intermediate checkpoints
        if (epoch+1) % args.log_interval == 0:
            torch.save(last_dict, f"{models_path}/checkpoint_{epoch}.pth")


if __name__ == '__main__':
    main()
