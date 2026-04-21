import os
import cv2
import yaml
import torch
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm

from f3.utils import num_params, plot_patched_features, setup_torch, ev_to_frames, log_dict
from f3.tasks.optical_flow.utils import EventFlow, EventFFFlow, get_dataloader_from_args_gtev, flow_viz_np, eval_flow


parser = argparse.ArgumentParser("Train a Optical Flow model on a dataset of events.")

parser.add_argument("--name", type=str, required=True, help="Name of the run.")
parser.add_argument("--useconfig", type=str, default=None, help="Path to the config file.")
parser.add_argument("--model", type=str, default="best", help="Model to load for evaluation.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the optical flow model")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")
parser.add_argument("--baseline", action="store_true", help="Use the baseline model.")
parser.add_argument("--save_preds", action="store_true", help="Save predictions during evaluation.")
parser.add_argument("--subsample", type=int, default=1, help="Subsampling factor for events (e.g., 2 means use 1/2 of events, randomly selected).")
parser.add_argument("--MVSEC_DT1", action="store_true", help="Scale to 45 Hz for MVSEC")
parser.add_argument("--MVSEC_DT4", action="store_true", help="Scale to 11.25 Hz for MVSEC")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = args.useconfig if args.useconfig else f"outputs/optical_flow/{args.name}/args.yml"
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


def build_valid_flow_mask(gt_flow, event_mask, data_name):
    # Keep only finite GT vectors with non-zero magnitude and event support.
    gt_finite = torch.isfinite(gt_flow).all(dim=-1)
    gt_nonzero = torch.norm(gt_flow, dim=-1) > 0
    valid_mask = gt_finite & gt_nonzero & event_mask
    if data_name == "mvsec":
        valid_mask[:, 193:] = 0
    return valid_mask


@torch.no_grad()
def test_fixed_time_optical_flow(args, model, val_loader, test_path, logger=None, save_preds=False):
    #! Technically just supports batch size 1 if multiple resolutions are used
    model.eval()
    results = {'1pe': torch.tensor([0.0]).cuda(), '2pe': torch.tensor([0.0]).cuda(), '3pe': torch.tensor([0.0]).cuda(),
               'aepe': torch.tensor([0.0]).cuda(), 'aae': torch.tensor([0.0]).cuda()}
    nsamples = torch.tensor([0.0]).cuda()

    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        # [(N,3) or (N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        events_flow, counts_flow, gt_flow, src_ofst_res = data

        events_flow, counts_flow, gt_flow = events_flow.cuda(), counts_flow.cuda(), gt_flow.cuda()

        if args.data == "mvsec":
            if args.MVSEC_DT1:
                gt_flow = gt_flow * 0.44444444444444444444444444
                logger.info("Rescaling to 45 Hz GT for MVSEC.")
            elif args.MVSEC_DT4:
                gt_flow = gt_flow * 1.777777777777777777777777778
                logger.info("Rescaling to 11.25 Hz GT for MVSEC.")
            else:
                logger.info("Using original 20 Hz GT for MVSEC.")

        if args.subsample > 1:
            keep_prob = 1.0 / args.subsample
            keep_mask = torch.rand(events_flow.shape[0], device=events_flow.device) < keep_prob

            # Avoid an empty event tensor for rare cases with aggressive subsampling.
            if not torch.any(keep_mask):
                keep_mask[torch.randint(events_flow.shape[0], (1,), device=events_flow.device)] = True

            events_flow = events_flow[keep_mask]

            split_masks = torch.split(keep_mask, counts_flow.tolist())
            counts_flow = torch.tensor(
                [int(mask.sum().item()) for mask in split_masks],
                dtype=counts_flow.dtype,
                device=counts_flow.device,
            )

        crop_params = torch.cat([
            src_ofst_res[:, :2],
            src_ofst_res[:, :2] + src_ofst_res[:, 2:]
        ], dim=1).int()

        flow_pred, ffflow = model(events_flow, counts_flow, cparams=crop_params)
        flow_pred, gt_flow = flow_pred.permute(0, 2, 3, 1), gt_flow.permute(0, 2, 3, 1) # (B, 2, H, W) -> (B, H, W, 2)

        if flow_pred.shape[1] != args.frame_sizes[1]:
            # pad flow_pred with 0s to the same size as the ground truth
            nrows = args.frame_sizes[1] - flow_pred.shape[1]
            flow_pred = torch.concat([flow_pred, torch.zeros(flow_pred.shape[0], nrows, flow_pred.shape[2], 2).cuda()], dim=1)

        event_frames_flow = ev_to_frames(events_flow, counts_flow, *args.frame_sizes).permute(0, 2, 1).cpu().numpy() # (B, H, W)
        event_mask = torch.from_numpy((event_frames_flow == 255)).to(flow_pred.device).bool()
        valid_mask = build_valid_flow_mask(gt_flow, event_mask, args.data)

        if valid_mask.sum() < 100:
            continue

        unmasked_flow_pred = flow_pred.clone()
        flow_pred = torch.where(valid_mask.unsqueeze(-1), flow_pred, torch.zeros_like(flow_pred))
        cur_results = eval_flow(flow_pred, gt_flow, valid_mask)
        for k, v in cur_results.items():
            results[k] += v
        nsamples += 1

        epe = torch.norm(flow_pred - gt_flow, dim=-1) * valid_mask

        if save_preds:
            logger.info(f"Saving predictions for batch: {idx}...")

            ffflow = ffflow.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
            for i in range(ffflow.shape[0]):
                flow_pred_rgb = flow_viz_np(unmasked_flow_pred[i].cpu().numpy())
                flow_gt_rgb = flow_viz_np(gt_flow[i].cpu().numpy(), norm=True)

                overlay_mask = valid_mask[i].detach().cpu().numpy().astype(np.uint8)

                overlay_image = flow_pred_rgb.copy()
                overlay_image *= overlay_mask[..., None]
                overlay_gt = flow_gt_rgb.copy()
                overlay_gt *= overlay_mask[..., None]

                ffflowpca, _ = plot_patched_features(ffflow[i], plot=False)
                ffflowpca = ffflowpca[..., ::-1]

                err_img = epe[i].clone().cpu().numpy()
                err_img = np.nan_to_num(err_img, nan=0.0, posinf=0.0, neginf=0.0)
                denom = err_img.max() - err_img.min()
                if denom > 0:
                    err_img = (err_img - err_img.min()) / denom
                else:
                    err_img = np.zeros_like(err_img)
                err_img = (err_img * 255).astype(np.uint8)
                err_img = cv2.applyColorMap(err_img, cv2.COLORMAP_JET)
                cv2.putText(err_img, f"Mean: {epe[valid_mask].mean():.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Std: {epe[valid_mask].std():.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Min: {epe[valid_mask].min():.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Max: {epe[valid_mask].max():.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(err_img, f"Flow Mean: {gt_flow[valid_mask].abs().mean():.2f} Std: {gt_flow[valid_mask].abs().std():.2f}",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imwrite(f"{test_path}/predictions/flow_{idx}_{i}.png", flow_pred_rgb)
                cv2.imwrite(f"{test_path}/predictions/overlay_{idx}_{i}.png", overlay_image)
                cv2.imwrite(f"{test_path}/predictions/ffflow_{idx}_{i}.png", ffflowpca)
                cv2.imwrite(f"{test_path}/predictions/epe_{idx}_{i}.png", err_img)
                cv2.imwrite(f"{test_path}/gt/flow_{idx}_{i}.png", flow_gt_rgb)
                cv2.imwrite(f"{test_path}/gt/overlay_{idx}_{i}.png", overlay_gt)
                cv2.imwrite(f"{test_path}/gt/eventsflow_{idx}_{i}.png", event_frames_flow[i])

    for k in results.keys():
        results[k] /= nsamples

    logger.info("#"*50)
    logger.info("Test: ")
    log_dict(logger, results)
    logger.info("#"*50)

    return results


def main():
    setup_torch(cudnn_benchmark=True)
    
    foldername = os.path.basename(confpath).split(".")[0]
    model_path = f"outputs/optical_flow/{args.name}/models"
    test_path = f"outputs/optical_flow/{args.name}/test/{args.model}/{foldername}"
    
    os.makedirs(test_path, exist_ok=True)
    for dir in ["predictions", "gt"]:
        os.makedirs(f"{test_path}/{dir}", exist_ok=True)
    shutil.copyfile(confpath, test_path + "/run.yml")

    logging.basicConfig(filename=f"{test_path}/exp.log", filemode="a",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Running prediction on: {args.name}")

    val_loader = get_dataloader_from_args_gtev(args, logger, shuffle=False, train=False)

    logger.info("#"*50)
    if not args.baseline:
        model = EventFFFlow(args.eventff["config"], flowhead_config=args.flowhead, return_loss=False)
        model.eventff = torch.compile(model.eventff)
    else:
        model = EventFlow(args.eventmodel, *args.frame_sizes, args.time_ctx // args.bucket,
                          flowhead_config=args.flowhead, return_loss=False)
        if args.compile:
            model.upchannel = torch.compile(model.upchannel)
    if args.compile:
        model.flowhead = torch.compile(model.flowhead)
    model.save_configs(model_path)

    logger.info(f"Feature Field + Optical Flow: {model}")
    logger.info(f"Trainable parameters in Optical Flow: {num_params(model.flowhead)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info("#"*50)

    last_dict = torch.load(f"{model_path}/{args.model}.pth", weights_only=False)
    model.load_state_dict(last_dict["model"], strict=True)
    last_epoch = last_dict.get("epoch", "N/A")
    last_loss = last_dict.get("loss", "N/A")
    del last_dict
    torch.cuda.empty_cache()
    logger.info(f"Loaded model from: {args.model}, Epoch: {last_epoch}, Loss: {last_loss}")

    test_fixed_time_optical_flow(args, model, val_loader, test_path, logger=logger, save_preds=args.save_preds)


if __name__ == '__main__':
    main()
