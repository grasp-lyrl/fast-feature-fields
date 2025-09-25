import os
import cv2
import yaml
import torch
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import colormaps as cm

from f3.utils import num_params, unnormalize_events, plot_patched_features, setup_torch, log_dict
from f3.tasks.depth.utils import (EventFFDepthAnythingV2, EventDepthAnythingV2,
                                      eval_depth, eval_disparity, get_disparity_image,
                                      get_depth_image, get_dataloader_from_args)


parser = argparse.ArgumentParser("Test a depth model on a dataset of events.")

parser.add_argument("--name", type=str, required=True, help="Name of the run.")
parser.add_argument("--useconfig", type=str, default=None, help="Path to the config file.")
parser.add_argument("--model", type=str, default="best", help="Model to load for evaluation.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the depth model")
parser.add_argument("--baseline", action="store_true", help="Use the baseline model.")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")
parser.add_argument("--save_preds", action="store_true", help="Save predictions and ground truth images.")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = args.useconfig if args.useconfig else f"outputs/monoculardepth/{args.name}/args.yml"
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


@torch.no_grad()
def test_fixed_time(args, model, test_loader, test_path, logger=None, save_preds=False):
    model.eval()
    maxi, mini = 0, 100
    cmap = cm.get_cmap('magma') if args.eval == "disparity" else cm.get_cmap('jet')
    if args.eval == "depth":
        results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(),
                   'd3': torch.tensor([0.0]).cuda(), 'abs_rel': torch.tensor([0.0]).cuda(),
                   'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(), 
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(),
                   'silog': torch.tensor([0.0]).cuda()}
    elif args.eval == "disparity":
        results = {'1pe': torch.tensor([0.0]).cuda(), '2pe': torch.tensor([0.0]).cuda(),
                   '3pe': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
                   'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(),
                   'silog': torch.tensor([0.0]).cuda()}
    
    eval_ = eval_depth if args.eval == "depth" else eval_disparity
    get_image_ = get_depth_image if args.eval == "depth" else get_disparity_image

    results_cutoff = {'10': torch.tensor([0.0]).cuda(),
                      '20': torch.tensor([0.0]).cuda(),
                      '30': torch.tensor([0.0]).cuda()}

    nsamples = torch.tensor([0.0]).cuda()
    for idx, data in tqdm(enumerate(test_loader), total=len(test_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        ff_events, event_counts, depth, src_ofst_res = data

        ff_events = ff_events.cuda()
        event_counts = event_counts.cuda()
        depth = depth.cuda().float() # (B,H,W)

        if not args.polarity[0]: ff_events = ff_events[..., :3]

        crop_params = torch.cat([
            src_ofst_res[0, :2],
            src_ofst_res[0, :2] + src_ofst_res[0, 2:]
        ], dim=0).int() # (4,)

        depth = depth[0, crop_params[0]:crop_params[2], crop_params[1]:crop_params[3]].unsqueeze(0) # (1,H,W)

        depth_pred, ff = model.infer_image(ff_events, event_counts, crop_params) # (H,W) & (C,H,W)
        #! Model always returns disparity, metric or scaleless

        if args.eval == "depth":
            depth = args.fb / depth # args.fb = focal length * baseline in pixel * meters
            depth_pred = args.fb / depth_pred # convert to depth from disparity
            valid_mask = (depth > args.min_depth) & (depth < args.max_depth)
        elif args.eval == "disparity":
            valid_mask = depth < args.max_disparity # don't want asburdly high disparities

        ff = ff.unsqueeze(0).permute(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
        depth_pred = depth_pred.unsqueeze(0)

        if valid_mask.sum() < 10:
            continue

        cur_results = eval_(depth_pred[valid_mask], depth[valid_mask])
        for k in cur_results.keys():
            results[k] += cur_results[k]
        for k in results_cutoff.keys():
            valid_mask_cutoff = valid_mask & (depth < int(k))
            results_cutoff[k] += torch.abs(depth - depth_pred)[valid_mask_cutoff].mean().item()
        nsamples += 1

        error = torch.abs(depth - depth_pred) * valid_mask

        if save_preds:
            logger.info(f"Saving predictions for batch: {idx}...")

            event_counts = torch.cumsum(torch.cat((torch.zeros(1), event_counts.cpu())), 0).to(torch.uint64)
            for i in range(ff.shape[0]):
                events = unnormalize_events(ff_events.cpu()[event_counts[i]:event_counts[i+1]].numpy(), args.frame_sizes) -\
                                            np.array([src_ofst_res[i, 1], src_ofst_res[i, 0]])

                depth_i = get_image_(depth[i], valid_mask[i], cmap)
                overlay = depth_i.copy()
                overlay[events[:, 1], events[:, 0]] = overlay[events[:, 1], events[:, 0]] // 2  + 96 # gray -> overlay

                depth_pred_i = get_image_(depth_pred[i], valid_mask[i], cmap)
                overlay_pred = depth_pred_i.copy()
                overlay_pred[events[:, 1], events[:, 0]] = overlay_pred[events[:, 1], events[:, 0]] // 2  + 96

                ffpca, _ = plot_patched_features(ff[i], plot=False)
                ffpca = ffpca[..., ::-1]

                err_img = error[i].clone().cpu().numpy()
                err_img = (err_img - err_img.min()) / (err_img.max() - err_img.min())
                err_img = (err_img * 255).astype(np.uint8)
                err_img = cv2.applyColorMap(err_img, cv2.COLORMAP_JET)
                cv2.putText(err_img, f"Mean: {error[valid_mask].mean():.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Std: {error[valid_mask].std():.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Min: {error[valid_mask].min():.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Max: {error[valid_mask].max():.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imwrite(f"{test_path}/gt/depth_{idx}_{i}.png", depth_i)
                cv2.imwrite(f"{test_path}/gt/overlay_{idx}_{i}.png", overlay)
                cv2.imwrite(f"{test_path}/predictions/error_{idx}_{i}.png", err_img)
                cv2.imwrite(f"{test_path}/predictions/feat_{idx}_{i}.png", ffpca)
                cv2.imwrite(f"{test_path}/predictions/depth_pred_{idx}_{i}.png", depth_pred_i)
                cv2.imwrite(f"{test_path}/predictions/overlay_pred_{idx}_{i}.png", overlay_pred)

    for k in results.keys():
        results[k] /= nsamples
    for k in results_cutoff.keys():
        results_cutoff[k] /= nsamples

    logger.info("#"*50)
    logger.info(f"Test: ")
    log_dict(logger, results)
    log_dict(logger, results_cutoff)
    logger.info("#"*50)

    return results


def main():
    setup_torch(cudnn_benchmark=True)

    foldername = os.path.basename(confpath).split(".")[0]
    model_path = f"outputs/monoculardepth/{args.name}/models"
    test_path = f"outputs/monoculardepth/{args.name}/test/{args.model}/{foldername}"

    os.makedirs(test_path, exist_ok=True)
    for dir in ["predictions", "gt"]:
        os.makedirs(f"{test_path}/{dir}", exist_ok=True)
    shutil.copyfile(confpath, f"{test_path}/run.yml")

    logging.basicConfig(filename=f"{test_path}/exp.log", filemode="a",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Running prediction on: {args.name}")

    val_loader = get_dataloader_from_args(args, logger, shuffle=False, train=False)

    logger.info("#"*50)
    if args.baseline:
        model = EventDepthAnythingV2(args.dav2_config, args.eventmodel, *args.frame_sizes, args.time_ctx // args.bucket)
    else:
        model = EventFFDepthAnythingV2(args.eventff["config"], args.dav2_config)
        if not args.compile:
            model.eventff = torch.compile(model.eventff)
    if args.compile:
        model = torch.compile(model)
    model.save_configs(test_path)

    logger.info(f"Feature Field + Monocular Depth: {model}")
    logger.info(f"Trainable parameters in Depth Anything V2: {num_params(model.dav2)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info("#"*50)

    last_dict = torch.load(f"{model_path}/{args.model}.pth", weights_only=True)
    model.load_state_dict(last_dict["model"], strict=False)
    epoch = last_dict["epoch"]
    results = last_dict["results"]
    del last_dict
    torch.cuda.empty_cache()
    logger.info(f"Loaded model from: {args.model} at epoch: {epoch}")
    log_dict(logger, results)

    test_fixed_time(args, model, val_loader, test_path, logger, save_preds=args.save_preds)


if __name__ == '__main__':
    main()
