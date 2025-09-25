import cv2
import torch
import wandb
import numpy as np
from tqdm import tqdm

from f3.utils import plot_patched_features, ev_to_frames
from f3.tasks.optical_flow.utils import flow_viz_np


@torch.no_grad()
def validate_fixed_time_optical_flow(args, model, val_loader, epoch,
                                     logger=None, save_preds=False):
    #! Technically just supports batch size 1 if multiple resolutions are used
    model.eval()
    val_loss, val_phto_loss, val_smooth_loss = 0, 0, 0
    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        events_old, events_new, events_flow, counts_old, counts_new, counts_flow, src_ofst_res = data

        events_old, events_new, events_flow = events_old.cuda(), events_new.cuda(), events_flow.cuda()
        counts_old, counts_new, counts_flow = counts_old.cuda(), counts_new.cuda(), counts_flow.cuda()

        cparams = torch.cat([
            src_ofst_res[:, :2],
            src_ofst_res[:, :2] + src_ofst_res[:, 2:]
        ], dim=1).int()

        flow_pred, ffflow, _, _, loss_dict = model(events_flow, counts_flow,
                                                   events_old, counts_old,
                                                   events_new, counts_new, cparams)

        loss = loss_dict["loss"]
        smoothness_loss = loss_dict["smoothness_loss"]
        photometric_loss = loss_dict["photometric_loss"]

        val_loss += loss.item()
        val_phto_loss += photometric_loss.item()
        val_smooth_loss += smoothness_loss.item()

        if idx % 20 == 0 and save_preds:
            logger.info(f"Saving predictions for epoch: {epoch} and batch: {idx}...")

            flow_pred = flow_pred.permute(0, 2, 3, 1).cpu().numpy() # (B, 2, H, W) -> (B, H, W, 2)
            ffflow = ffflow.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)

            event_frames_flow = ev_to_frames(events_flow, counts_flow, *args.frame_sizes).permute(0, 2, 1).cpu().numpy() # (B, H, W)
            event_frames_old = ev_to_frames(events_old, counts_old, *args.frame_sizes).permute(0, 2, 1).cpu().numpy() # (B, H, W)
            event_frames_new = ev_to_frames(events_new, counts_new, *args.frame_sizes).permute(0, 2, 1).cpu().numpy() # (B, H, W)
            shift_frames = np.stack([event_frames_old, np.zeros_like(event_frames_old), event_frames_new], axis=-1) # (B, H, W, 3)
            for i in range(ffflow.shape[0]):
                flow_pred_rgb = flow_viz_np(flow_pred[i])

                overlay_image = flow_pred_rgb.copy()
                if overlay_image.shape[0] != event_frames_flow[i].shape[0]: # if some pixels are removed
                    nrows = event_frames_flow[i].shape[0] - overlay_image.shape[0]
                    overlay_image = np.pad(overlay_image, ((0, nrows), (0, 0), (0, 0)), mode="constant", constant_values=0)

                mask = (event_frames_flow[i] == 255).astype(np.uint8)
                overlay_image *= mask[..., None]

                ffflowpca, _ = plot_patched_features(ffflow[i], plot=False)
                ffflowpca = ffflowpca[..., ::-1]

                cv2.imwrite(f"outputs/optical_flow/{args.name}/predictions/flow_{epoch}_{idx}_{i}.png", flow_pred_rgb)
                cv2.imwrite(f"outputs/optical_flow/{args.name}/predictions/overlay_{epoch}_{idx}_{i}.png", overlay_image)
                cv2.imwrite(f"outputs/optical_flow/{args.name}/predictions/featflow_{epoch}_{idx}_{i}.png", ffflowpca)
                cv2.imwrite(f"outputs/optical_flow/{args.name}/training_events/eventsflow_{epoch}_{idx}_{i}.png", event_frames_flow[i])
                cv2.imwrite(f"outputs/optical_flow/{args.name}/training_events/shifts_{epoch}_{idx}_{i}.png", shift_frames[i])

    val_loss /= len(val_loader)
    val_phto_loss /= len(val_loader)
    val_smooth_loss /= len(val_loader)

    logger.info("#"*50)
    logger.info(f"Validation: Epoch: {epoch}, Loss: {val_loss}, " +\
                f"Photometric Loss: {val_phto_loss}, Smoothness Loss: {val_smooth_loss}")
    logger.info("#"*50)

    if args.wandb:
        wandb.log({"val_loss": val_loss, "val_photometric_loss": val_phto_loss,
                   "val_smoothness_loss": val_smooth_loss, "epoch": epoch})

    return val_loss, val_phto_loss, val_smooth_loss
