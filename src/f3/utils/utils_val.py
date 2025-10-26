import cv2
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy

from f3.utils import (display_predicted_shifts_frames, smooth_time_weighted_rgb_encoding,
                      ev_to_grid, ev_to_frames, tp_tn_fp_fn, acc_f1, get_crop_targets, crop_and_resize_targets)


@torch.no_grad()
def validate_fixed_time(args, eff, val_loader, epoch,
                        accelerator=None, logger=None, save_preds=False):
    eff.eval()
    val_loss, val_acc, val_f1 = 0, 0, 0
    pred_frame_size = deepcopy(args.frame_sizes) + [args.time_pred // args.bucket]

    # Initialization for cropping and resizing
    crop_resize_training = getattr(args, 'random_crop_resize', None)
    if crop_resize_training is not None:
        crop_size, ctx_out_resolution, pred_out_resolution, pred_frame_size = get_crop_targets(args)

    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader), disable=not accelerator.is_local_main_process):
        ff_events, pred_events, ff_counts, pred_counts, valid_mask = data # (N,3) or (N,4), (B), (B,W,H,2) or (B,W,H,T,2) #! T: max prediction time bins time_pred//bucket

        if not args.polarity[0]: ff_events = ff_events[..., :3] # (B,N,4) -> (B,N,3)
        if not args.polarity[1]: pred_events = pred_events[..., :3]

        if crop_resize_training is not None:
            ff_events, pred_events, ff_counts, pred_counts, valid_mask = crop_and_resize_targets(
                args, crop_size, ff_events, pred_events, ff_counts, pred_counts, valid_mask,
                ctx_out_resolution, pred_out_resolution, pred_frame_size
            )

        pred_event_grid = ev_to_grid(pred_events, pred_counts, *pred_frame_size)

        predlogits, loss = eff(ff_events, ff_counts, pred_event_grid, valid_mask) # (B,N,3) -> (B,W,H,1) or (B,W,H,T)

        if idx % 20 == 0 and save_preds:
            event_frames = ev_to_frames(ff_events, ff_counts, *pred_frame_size[:2])
            event_frames = accelerator.gather_for_metrics(event_frames).cpu().numpy()
        predlogits, pred_event_grid, loss = accelerator.gather_for_metrics((predlogits, pred_event_grid, loss))

        if accelerator.is_local_main_process:
            val_loss += loss.mean().item()

            predictions = torch.sigmoid(predlogits)
            acc, f1 = acc_f1(*tp_tn_fp_fn(predictions, pred_event_grid))
            val_acc += acc
            val_f1 += f1

            if idx % 20 == 0 and save_preds:
                logger.info(f"Saving predictions for epoch: {epoch} and batch: {idx}...")

                # predictions are in one of the following formats: (B,W,H,T), (B,W,H,T,2)
                # We dont handle plotting polarity
                pred_full_event_volume = smooth_time_weighted_rgb_encoding((predictions > 0.5).cpu().numpy().astype(np.uint8)) # (B,W,H,3)
                event_shifts = np.zeros_like(pred_full_event_volume)
                event_shifts[..., 0] = event_frames
                event_shifts[pred_event_grid.any(-1).cpu().numpy(), 2] = 255
                
                pred_full_event_volume = pred_full_event_volume.transpose(0, 2, 1, 3)
                event_shifts = event_shifts.transpose(0, 2, 1, 3)

                for i in range(predlogits.shape[0]):
                    cv2.imwrite(f"outputs/{args.name}/predictions/predicted_{epoch}_full_{idx}_{i}.png", pred_full_event_volume[i])
                    display_predicted_shifts_frames(event_shifts[i, ..., 0], pred_full_event_volume[i], f"outputs/{args.name}/predictions/shift_{epoch}_{idx}_{i}")
                    cv2.imwrite(f"outputs/{args.name}/training_events/ff_{epoch}_{idx}_{i}.png", event_shifts[i,...,0])
                    cv2.imwrite(f"outputs/{args.name}/training_events/pos_{epoch}_{idx}_{i}.png", event_shifts[i,...,2])
                    cv2.imwrite(f"outputs/{args.name}/training_events/shift_{epoch}_{idx}_{i}.png", event_shifts[i])

    if accelerator.is_local_main_process:
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)

        logger.info("#"*50)
        logger.info(f"Validation: Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}, F1: {val_f1}")
        logger.info("#"*50)

        if args.wandb:
            accelerator.log({"val_acc": val_acc, "val_loss": val_loss, "val_f1": val_f1, "epoch": epoch})

    eff.train()
    return val_loss, val_acc, val_f1
