import os
import cv2
import h5py
import yaml
import torch
import shutil
import imageio
import logging
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from matplotlib import colormaps as cm

from f3.utils import BaseExtractor, num_params, plot_patched_features, setup_torch, ev_to_frames, log_dict
from f3.tasks.depth.utils import EventDepthAnythingV2, EventFFDepthAnythingV2, get_disparity_image


parser = argparse.ArgumentParser("Test a Monolcular Disparity model on DSEC timestamps. Generates the output directory with the 16bit disparity predictions.")

parser.add_argument("--name", type=str, required=True, help="Name of the run.")
parser.add_argument("--model", type=str, default="best", help="Model to load for evaluation.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the optical flow model")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")

# Eval args
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--baseline", action="store_true", help="Baseline")
parser.add_argument("--datapath", type=str, help="Path to the dataset")
parser.add_argument("--test_disparity_timestamps", type=str, help="Path to the timestamps folder, after extracting the zip")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = f"outputs/monoculardepth/{args.name}/args.yml"
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


@torch.no_grad()
def test_fixed_time_monoculardepth(args, model, event_data, eval_timestamps, save_path, logger=None):
    #! Technically just supports batch size 1 if multiple resolutions are used
    model.eval()
    cmap = cm.get_cmap('magma')
    nsamples = eval_timestamps.shape[0]
    for idx in tqdm(range(nsamples)):
        ts, file_no = eval_timestamps[idx]
        t0 = ts + args.time_ctx // 2
        t0 = t0 if t0 % 20 == 0 else (t0 // 20 + 1) * 20 # get it to be a multiple of 20

        events_depth, counts_depth = event_data.get_ctx_fixedtime(t0) # [ts_start - time_ctx//2, ts_start + time_ctx//2]
        counts_depth = torch.tensor([counts_depth], dtype=torch.int32)
        events_depth, counts_depth = events_depth.cuda(), counts_depth.cuda()

        ofsts = [(args.frame_sizes[1] - 480) // 2, (args.frame_sizes[0] - 640) // 2]
        cparams = torch.tensor([ofsts[0], ofsts[1], ofsts[0] + 480, ofsts[1] + 640], dtype=torch.int32)

        depth_pred, ff = model.infer_image(events_depth, counts_depth, cparams) # (H, W) & (C, H, W)
        ff = ff.permute(1, 2, 0) # (H, W, C)

        if args.debug:
            event_frames_depth = ev_to_frames(events_depth, counts_depth, *args.frame_sizes).permute(0, 2, 1).cpu().numpy()[0] # (B, H, W)
            event_frames_depth = event_frames_depth[cparams[0]:cparams[2], cparams[1]:cparams[3]]
            event_mask = (event_frames_depth == 255).astype(np.uint8)

            depth_pred_rgb = get_disparity_image(depth_pred, torch.ones_like(depth_pred, dtype=bool), cmap)
            overlay_image = depth_pred_rgb.copy()
            overlay_image *= event_mask[..., None]

            ffflowpca, _ = plot_patched_features(ff, plot=False)
            ffflowpca = ffflowpca[..., ::-1]

        file_name = str(file_no).zfill(6) + ".png"

        if args.debug:
            cv2.imwrite(f"{save_path}/depth_{file_name}", depth_pred_rgb)
            cv2.imwrite(f"{save_path}/overlay_{file_name}", overlay_image)
            cv2.imwrite(f"{save_path}/ffdepth_{file_name}", ffflowpca)
            cv2.imwrite(f"{save_path}/eventsdepth_{file_name}", event_frames_depth)

        depth_pred = torch.clamp(depth_pred, 6.8203125, 76.87890625)
        depth_pred = (depth_pred * 256).cpu().numpy().astype(np.uint16)
        imageio.imwrite(f"{save_path}/{file_name}", depth_pred, format="PNG-FI")


def main():
    setup_torch(cudnn_benchmark=True)

    model_path = f"outputs/monoculardepth/{args.name}/models"
    tsfolder = args.test_disparity_timestamps
    data_path = Path(args.datapath)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(confpath, output_dir + "/run.yml")

    logging.basicConfig(filename=f"{output_dir}/exp.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Running prediction on: {args.name}")

    logger.info("#"*50)
    if args.baseline:
        model = EventDepthAnythingV2(args.dav2_config, args.eventmodel,
                                     *args.frame_sizes, args.time_ctx // args.bucket)
        if args.compile:
            model = torch.compile(model, fullgraph=False)
    else:
        model = EventFFDepthAnythingV2(args.eventff["config"], args.dav2_config)
        if not args.compile:
            model.eventff = torch.compile(model.eventff, fullgraph=False)
        else:
            model = torch.compile(model, fullgraph=False)
    model.save_configs(output_dir)

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

    # list of files in the folder using Path
    files = list(Path(tsfolder).rglob("*.csv"))
    dataset_names = [f.stem for f in files]
    event_paths = [data_path / dataset_name / "events/left/events.h5" for dataset_name in dataset_names]
    timestamps_50khz = [data_path / dataset_name / f"50khz_{dataset_name}.npy" for dataset_name in dataset_names]
    eval_timestamp_paths = [str(f) for f in files]

    for dataset_name, event_path, ts50khz, eval_timestamp_path in zip(
        dataset_names, event_paths, timestamps_50khz, eval_timestamp_paths
    ):
        logger.info(f"Testing on: {dataset_name}")
        data_output_path = f"{output_dir}/{dataset_name}"
        os.makedirs(data_output_path, exist_ok=True)
        event_extractor = BaseExtractor(event_path, ts50khz, args.frame_sizes[0], args.frame_sizes[1],
                                        args.time_ctx, None, args.bucket, args.max_numevents_ctx,
                                        randomize_ctx=False, camera="left", dtype="dsec")
        eval_timestamps = np.genfromtxt(eval_timestamp_path, delimiter=",", dtype=np.uint64, skip_header=1)
        eval_timestamps[:, 0] -= h5py.File(event_path, "r")['t_offset'][()].astype(np.uint64)
        test_fixed_time_monoculardepth(args, model, event_extractor, eval_timestamps, data_output_path, logger=logger)

if __name__ == '__main__':
    main()
