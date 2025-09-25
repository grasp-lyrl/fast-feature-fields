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

from f3 import init_event_model, load_weights_ckpt
from f3.utils import setup_torch, ev_to_frames, plot_patched_features, smooth_time_weighted_rgb_encoding, log_dict

from f3.tasks.depth.utils import init_depth_model, load_depth_weights
from f3.tasks.optical_flow.utils import init_flow_model, load_flow_weights
from f3.tasks.segmentation.utils import init_segmentation_model, load_segmentation_weights

from f3.tasks.depth.utils import get_disparity_image
from f3.tasks.optical_flow.utils import get_dataloader_from_args_imevev, flow_viz_np
from f3.tasks.segmentation.utils import cityscapes_palette

parser = argparse.ArgumentParser("Visualize outputs from all tasks.")
parser.add_argument("--conf", type=str, required=True, help="Yaml file containing paths to model configs and weights.")
parser.add_argument("--output", type=str, required=True, help="Output folder")

args = parser.parse_args()

with open(args.conf, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    setattr(args, key, value)


@torch.no_grad()
def runall(args, loader, eventff_model, seg_model, optflow_model, depth_model):
    cmap = cm.get_cmap('magma')
    crop = lambda img, cparam: img[cparam[0]:cparam[2], cparam[1]:cparam[3]]

    for idx, data in tqdm(enumerate(loader), total=len(loader)):
        ff_events1, _, event_counts1, _, img, src_ofst_res = data

        if not args.polarity[0]: ff_events1 = ff_events1[..., :3]

        ff_events1, event_counts1 = ff_events1.cuda(), event_counts1.cuda()

        crop_params = torch.cat([
            src_ofst_res[:, :2],
            src_ofst_res[:, :2] + src_ofst_res[:, 2:]
        ], dim=1).int()

        event1_img = ev_to_frames(ff_events1.cpu().numpy(), event_counts1.cpu().numpy(), *(args.frame_sizes))[0].T
        event1_img = crop(event1_img, crop_params[0])
        cv2.imwrite(f"{args.output}/events/event1_{idx}.png", event1_img)
        cv2.imwrite(f"{args.output}/images/image_{idx}.png", (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        if eventff_model:
            futureev_logits, ff = eventff_model(ff_events1, event_counts1)

            futureev = torch.sigmoid(futureev_logits) # B, W, H, T
            futureev_vol = smooth_time_weighted_rgb_encoding((futureev > 0.5).cpu().numpy().astype(np.uint8)).transpose(0,2,1,3)
            # make a frame from futureev if any time step has more than 1 event
            futureev_frame = (futureev[0] > 0.5).any(-1).cpu().numpy().astype(np.uint8).T * 255
            ff = ff.permute(0, 2, 1, 3) # (B, W, H, C) -> (B, H, W, C)
            ffpca = plot_patched_features(ff[0], plot=False)[0][..., ::-1]
            cv2.imwrite(f"{args.output}/events/futureev_frame_{idx}.png", crop(futureev_frame, crop_params[0]))
            cv2.imwrite(f"{args.output}/events/futureev_{idx}.png", crop(futureev_vol[0], crop_params[0]))
            cv2.imwrite(f"{args.output}/events/feat_{idx}.png", crop(ffpca, crop_params[0]))

        if optflow_model:    
            flow_pred, _ = optflow_model(ff_events1, event_counts1, crop_params)
            flow_pred = flow_pred.permute(0, 2, 3, 1).cpu().numpy() # (B, 2, H, W) -> (B, H, W, 2)
            flow_pred = crop(flow_pred[0], crop_params[0])

            flow_img = flow_viz_np(flow_pred)
            flow_mask = (event1_img == 255).astype(np.uint8)
            overlay_flow_img = flow_img.copy()
            overlay_flow_img *= flow_mask[..., None]
            cv2.imwrite(f"{args.output}/optical_flow/flow_{idx}.png", flow_img)
            cv2.imwrite(f"{args.output}/optical_flow/overlay_flow_{idx}.png", overlay_flow_img)

        if depth_model:
            depth_pred, _ = depth_model.infer_image(ff_events1, event_counts1, crop_params[0])

            depth_mask = torch.ones_like(depth_pred, dtype=torch.bool)
            depth_img = get_disparity_image(depth_pred, depth_mask, cmap)
            overlay_depth_img = depth_img.copy()
            overlay_depth_img[event1_img == 255] = 0.5 * overlay_depth_img[event1_img == 255] + 64
            cv2.imwrite(f"{args.output}/monoculardepth/depth_{idx}.png", depth_img)
            cv2.imwrite(f"{args.output}/monoculardepth/overlay_depth_{idx}.png", overlay_depth_img)

        if seg_model:
            seg_pred, _ = seg_model(ff_events1, event_counts1, crop_params)
            seg_pred = seg_pred.argmax(1).cpu().numpy()

            seg_img = cityscapes_palette(seg_model.num_labels)[seg_pred[0]].astype(np.uint8)
            overlay_seg_img = seg_img.copy()
            overlay_seg_img[event1_img == 255] = 0.5 * overlay_seg_img[event1_img == 255] + 64
            cv2.imwrite(f"{args.output}/segmentation/seg_{idx}.png", seg_img)
            cv2.imwrite(f"{args.output}/segmentation/overlay_seg_{idx}.png", overlay_seg_img)


def main():
    setup_torch(cudnn_benchmark=True)

    os.makedirs(args.output, exist_ok=True)
    for dir in ["monoculardepth", "segmentation", "optical_flow", "events", "images"]:
        os.makedirs(f"{args.output}/{dir}", exist_ok=True)
    shutil.copyfile(args.conf, args.output + "/run.yml")

    logging.basicConfig(filename=f"{args.output}/exp.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)

    loader = get_dataloader_from_args_imevev(args, logger, shuffle=False, train=False)

    eventff_config = getattr(args, 'eventff', None)
    segmentation_config = getattr(args, 'segmentation', None)
    optical_flow_config = getattr(args, 'optical_flow', None)
    monoculardepth_config = getattr(args, 'monoculardepth', None)

    ####### Load Segmentation Model #######
    seg_model = None
    if segmentation_config is not None:
        seg_model = init_segmentation_model(segmentation_config["config"]).cuda()
        seg_model.eventff = torch.compile(seg_model.eventff, fullgraph=False)
        epoch, loss, acc, miou = load_segmentation_weights(seg_model, segmentation_config["ckpt"])
        logger.info(f"Loaded Segmentation ckpt from {segmentation_config['ckpt']}. " +\
                    f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}, MIoU: {miou}")
        seg_model.eval()

    ####### Load Optical Flow Model #######
    optflow_model = None
    if optical_flow_config is not None:
        optflow_model = init_flow_model(optical_flow_config["config"]).cuda()
        optflow_model.eventff = torch.compile(optflow_model.eventff, fullgraph=False)
        optflow_model.flowhead = torch.compile(optflow_model.flowhead, fullgraph=False)
        epoch, loss = load_flow_weights(optflow_model, optical_flow_config["ckpt"])
        logger.info(f"Loaded Optical Flow ckpt from {optical_flow_config['ckpt']}. " +\
                    f"Epoch: {epoch}, Loss: {loss}")
        optflow_model.eval()
    
    ####### Load Event Feature Model #######
    eventff_model = None
    if eventff_config is not None:
        eventff_model = init_event_model(eventff_config["config"], return_feat=True, return_logits=True).cuda()
        eventff_model = torch.compile(eventff_model, fullgraph=False)
        epoch, loss, acc = load_weights_ckpt(eventff_model, eventff_config["ckpt"])
        logger.info(f"Loaded Event Feature ckpt from {eventff_config['ckpt']}. " +\
                    f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}")
        eventff_model.eval()

    ####### Load Monocular Depth Model #######
    depth_model = None
    if monoculardepth_config is not None:
        depth_model = init_depth_model(monoculardepth_config["config"]).cuda()
        depth_model.eventff = torch.compile(depth_model.eventff, fullgraph=False)
        epoch, results = load_depth_weights(depth_model, monoculardepth_config["ckpt"])
        logger.info(f"Loaded Monocular Depth ckpt from {monoculardepth_config['ckpt']}.")
        log_dict(logger, results)
        depth_model.eval()

    runall(args, loader, eventff_model, seg_model, optflow_model, depth_model)


if __name__ == "__main__":
    main()
