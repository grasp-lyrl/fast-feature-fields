import cv2
import wandb
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import cm

from f3.utils import unnormalize_events, log_dict
from f3.tasks.depth.utils import eval_disparity, get_disparity_image


@torch.no_grad()
def validate_fixed_time_disparity(args, model, val_loader, loss_fn, epoch, logger=None, save_preds=False):
    model.eval()
    cmap = cm.get_cmap('magma')
    results = {'1pe': torch.tensor([0.0]).cuda(), '2pe': torch.tensor([0.0]).cuda(),
               '3pe': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
               'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(),
               'silog': torch.tensor([0.0]).cuda(), loss_fn.name: torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()
    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        ff_events, event_counts, disparity, src_ofst_res = data

        #! Expecting batch size of 1 for validation since we have varying resolutions in datasets !!!!
        ff_events = ff_events.cuda()
        event_counts = event_counts.cuda()
        disparity = disparity.cuda().float() # (B,H,W)

        if not args.polarity[0]: ff_events = ff_events[..., :3]

        crop_params = torch.cat([
            src_ofst_res[0, :2],
            src_ofst_res[0, :2] + src_ofst_res[0, 2:]
        ], dim=0).int() # (4,)

        disparity = disparity[0, crop_params[0]:crop_params[2], crop_params[1]:crop_params[3]].unsqueeze(0) # (1,H,W)

        disparity_pred = model.infer_image(ff_events, event_counts, crop_params)[0].unsqueeze(0) # (1,N,3) -> (1,H,W)

        valid_mask = disparity < args.max_disparity # don't want asburdly high disparities

        if valid_mask.sum() < 10:
            continue

        cur_results = eval_disparity(disparity_pred[valid_mask], disparity[valid_mask])
        for k in cur_results.keys():
            results[k] += cur_results[k]
        results[loss_fn.name] += loss_fn(disparity_pred, disparity, valid_mask).item()
        nsamples += 1

        if idx % 20 == 0 and save_preds:
            logger.info(f"Saving predictions for epoch: {epoch} and batch: {idx}...")

            ff_events = ff_events[..., :3] # (B,N,3)
            event_counts = torch.cumsum(torch.cat((torch.zeros(1), event_counts.cpu())), 0).to(torch.uint64)
            for i in range(disparity_pred.shape[0]):
                events = unnormalize_events(ff_events.cpu()[event_counts[i]:event_counts[i+1]].numpy(), args.frame_sizes) -\
                                            np.array([src_ofst_res[i, 1], src_ofst_res[i, 0]])

                disparity_i = get_disparity_image(disparity[i], valid_mask[i], cmap)
                overlay = disparity_i.copy()
                overlay[events[:, 1], events[:, 0]] = overlay[events[:, 1], events[:, 0]] // 2  + 96 # gray -> overlay

                disparity_pred_i = get_disparity_image(disparity_pred[i], torch.ones_like(disparity_pred[i], dtype=torch.bool), cmap)
                overlay_pred = disparity_pred_i.copy()
                overlay_pred[events[:, 1], events[:, 0]] = overlay_pred[events[:, 1], events[:, 0]] // 2  + 96

                cv2.imwrite(f"outputs/monoculardepth/{args.name}/training_events/disparity_{idx}_{i}.png", disparity_i)
                cv2.imwrite(f"outputs/monoculardepth/{args.name}/training_events/overlay_{epoch}_{idx}_{i}.png", overlay)
                cv2.imwrite(f"outputs/monoculardepth/{args.name}/predictions/disparity_pred_{epoch}_{idx}_{i}.png", disparity_pred_i)
                cv2.imwrite(f"outputs/monoculardepth/{args.name}/predictions/overlay_pred_{epoch}_{idx}_{i}.png", overlay_pred)

    for k in results.keys():
        results[k] /= nsamples

    logger.info("#"*50)
    logger.info(f"Validation: Epoch: {epoch}")
    log_dict(logger, results)
    logger.info("#"*50)

    if args.wandb:
        wandb_dict = {f"val_{k}": v.item() for k, v in results.items()}
        wandb_dict["epoch"] = epoch
        wandb.log(wandb_dict)
    return results
