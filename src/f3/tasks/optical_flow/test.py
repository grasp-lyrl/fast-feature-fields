import os
import cv2
import yaml
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch._dynamo

from f3.utils import num_params, unnormalize_events, plot_patched_features, setup_torch
from f3.tasks.optical_flow.utils import EventFFFlow, get_dataloader_from_args_evev, flow_viz_np


parser = argparse.ArgumentParser("Train a Optical Flow model on a dataset of events.")

parser.add_argument("--name", type=str, required=True, help="Name of the run.")
parser.add_argument("--useconfig", type=str, default=None, help="Path to the config file.")
parser.add_argument("--model", type=str, default="best", help="Model to load for evaluation.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the optical flow model")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = args.useconfig if args.useconfig else f"outputs/optical_flow/{args.name}/args.yml"
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


@torch.no_grad()
def test_fixed_time_optical_flow(args, model, val_loader, test_path, logger=None, save_preds=False):
    #! Technically just supports batch size 1 if multiple resolutions are used
    model.eval()
    test_loss, test_phto_loss, test_smooth_loss = 0, 0, 0
    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
        # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        ff_events1, ff_events2, event_counts1, event_counts2, src_ofst_res = data
        
        if not args.polarity[0]:
            ff_events1 = ff_events1[..., :3]
            ff_events2 = ff_events2[..., :3]

        ff_events1, ff_events2 = ff_events1.cuda(), ff_events2.cuda()
        event_counts1, event_counts2 = event_counts1.cuda(), event_counts2.cuda()

        crop_params = torch.cat([
            src_ofst_res[:, :2],
            src_ofst_res[:, :2] + src_ofst_res[:, 2:]
        ], dim=1).int()

        ff1, flow_pred, ff2, loss_dict = model(ff_events1, event_counts1, ff_events2, event_counts2, crop_params)

        loss = loss_dict["loss"]
        smoothness_loss = loss_dict["smoothness_loss"]
        photometric_loss = loss_dict["photometric_loss"]

        test_loss += loss.item()
        test_phto_loss += photometric_loss.item()
        test_smooth_loss += smoothness_loss.item()

        flow_pred = flow_pred.permute(0, 2, 3, 1).cpu().numpy() # (B, 2, H, W) -> (B, H, W, 2)
        ff1 = ff1.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        ff2 = ff2.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        H, W = ff1.shape[1], ff1.shape[2]

        if save_preds:
            logger.info(f"Saving predictions for batch: {idx}...")

            event_counts1 = torch.cumsum(torch.cat((torch.zeros(1), event_counts1.cpu())), 0).to(torch.uint64)
            event_counts2 = torch.cumsum(torch.cat((torch.zeros(1), event_counts2.cpu())), 0).to(torch.uint64)
            event_frame1 = np.zeros((H, W), dtype=np.uint8)
            event_frame2 = np.zeros_like(event_frame1)
            for i in range(ff1.shape[0]):
                events1 = unnormalize_events(ff_events1.cpu()[event_counts1[i]:event_counts1[i+1]].numpy(), args.frame_sizes) -\
                          np.array([src_ofst_res[i, 1], src_ofst_res[i, 0]])
                events2 = unnormalize_events(ff_events2.cpu()[event_counts2[i]:event_counts2[i+1]].numpy(), args.frame_sizes) -\
                          np.array([src_ofst_res[i, 1], src_ofst_res[i, 0]])
                event_frame1[events1[:, 1], events1[:, 0]] = 255 # white -> context events
                event_frame2[events2[:, 1], events2[:, 0]] = 255 # white -> context events
                flow_pred_rgb = flow_viz_np(flow_pred[i])

                overlay_image = flow_pred_rgb.copy()
                mask = np.logical_or(event_frame1 == 255, event_frame2 == 255).astype(np.uint8)
                overlay_image *= mask[..., None]

                ff1pca, _ = plot_patched_features(ff1[i], plot=False)
                ff1pca = ff1pca[..., ::-1]

                ff2pca, _ = plot_patched_features(ff2[i], plot=False)
                ff2pca = ff2pca[..., ::-1]

                cv2.imwrite(f"{test_path}/predictions/flow_{idx}_{i}.png", flow_pred_rgb)
                cv2.imwrite(f"{test_path}/predictions/overlay_{idx}_{i}.png", overlay_image)
                cv2.imwrite(f"{test_path}/predictions/feat1_{idx}_{i}.png", ff1pca)
                cv2.imwrite(f"{test_path}/predictions/feat2_{idx}_{i}.png", ff2pca)
                cv2.imwrite(f"{test_path}/training_events/events1_{idx}_{i}.png", event_frame1)
                cv2.imwrite(f"{test_path}/training_events/events2_{idx}_{i}.png", event_frame2)

    test_loss /= len(val_loader)
    test_phto_loss /= len(val_loader)
    test_smooth_loss /= len(val_loader)

    logger.info("#"*50)
    logger.info(f"Test: Loss: {test_loss}, Photometric Loss: {test_phto_loss}, Smoothness Loss: {test_smooth_loss}")
    logger.info("#"*50)

    return test_loss, test_phto_loss, test_smooth_loss


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

    val_loader = get_dataloader_from_args_evev(args, logger, shuffle=False, train=False)

    logger.info("#"*50)
    model = EventFFFlow(args.eventff["config"], pyramids=args.pyramids, alpha=args.alpha, return_loss=True)
    if not args.compile:
        model.eventff = torch.compile(model.eventff)
    else:
        model = torch.compile(model)
    model.load_weights(args.eventff["ckpt"])
    model.save_configs(test_path)

    logger.info(f"Feature Field + Optical Flow: {model}")
    logger.info(f"Trainable parameters in Optical Flow: {num_params(model.flowhead)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info("#"*50)

    last_dict = torch.load(f"{model_path}/{args.model}.pth", weights_only=False)
    model.load_state_dict(last_dict["model"], strict=False)
    last_epoch = last_dict["epoch"]
    last_loss = last_dict["loss"]
    del last_dict
    torch.cuda.empty_cache()
    logger.info(f"Loaded model from: {args.model}, Epoch: {last_epoch}, Loss: {last_loss}")

    test_fixed_time_optical_flow(args, model, val_loader, test_path, logger=logger, save_preds=True)


if __name__ == '__main__':
    main()
