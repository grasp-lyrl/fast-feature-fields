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

from f3.utils import BaseExtractor, num_params, plot_patched_features, setup_torch, ev_to_frames
from f3.tasks.optical_flow.utils import EventFlow, EventFFFlow, flow_viz_np


parser = argparse.ArgumentParser("Test a Optical Flow model on DSEC timestamps. Generates the output directory with the 16bit flow predictions.")

parser.add_argument("--name", type=str, required=True, help="Name of the run.")
parser.add_argument("--model", type=str, default="best", help="Model to load for evaluation.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the optical flow model")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")
parser.add_argument("--baseline", action="store_true", help="Use the baseline model.")

# Eval args
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument("--datapath", type=str, help="Path to the dataset")
parser.add_argument("--test_forward_optical_flow_timestamps", type=str, help="Path to the timestamps folder, after extracting the zip")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = f"outputs/optical_flow/{args.name}/args.yml"
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


@torch.no_grad()
def test_fixed_time_optical_flow(args, model, event_data, eval_timestamps, save_path, logger=None):
    #! Technically just supports batch size 1 if multiple resolutions are used
    model.eval()
    nsamples = eval_timestamps.shape[0]
    for idx in tqdm(range(nsamples)):
        ts_start, _, file_no = eval_timestamps[idx]
        t0 = ts_start + args.time_ctx // 2
        t0 = t0 if t0 % 20 == 0 else (t0 // 20 + 1) * 20 # get it to be a multiple of 20

        events_flow, counts_flow = event_data.get_ctx_fixedtime(t0) # [ts_start - time_ctx//2, ts_start + time_ctx//2]
        counts_flow = torch.tensor([counts_flow], dtype=torch.int32)
        events_flow, counts_flow = events_flow.cuda(), counts_flow.cuda()

        cparams = torch.tensor([[0, 0, 450, 640]], dtype=torch.int32)

        flow_pred, ffflow = model(events_flow, counts_flow, cparams=cparams)
        flow_pred = flow_pred.permute(0, 2, 3, 1)[0] # (1, 2, H, W) -> (H, W, 2)
        ffflow = ffflow.permute(0, 2, 3, 1)[0] # (1, C, H, W) -> (H, W, C)

        dsec_flow = flow_pred.cpu().numpy()

        nrows = 30
        dsec_flow = np.pad(dsec_flow, ((0, nrows), (0, 0), (0, 0)), mode="constant", constant_values=0)

        if args.debug:
            event_frames_flow = ev_to_frames(events_flow, counts_flow, *args.frame_sizes).permute(0, 2, 1).cpu().numpy()[0] # (B, H, W)
            event_mask = (event_frames_flow == 255).astype(np.uint8)

            flow_pred_rgb = flow_viz_np(flow_pred.cpu().numpy())
            overlay_image = flow_pred_rgb.copy()

            overlay_image = np.pad(overlay_image, ((0, nrows), (0, 0), (0, 0)), mode="constant", constant_values=0)
            overlay_image *= event_mask[..., None]

            ffflowpca, _ = plot_patched_features(ffflow, plot=False)
            ffflowpca = ffflowpca[..., ::-1]

        file_name = str(file_no).zfill(6) + ".png"

        if args.debug:
            cv2.imwrite(f"{save_path}/flow_{file_name}", flow_pred_rgb)
            cv2.imwrite(f"{save_path}/overlay_{file_name}", overlay_image)
            cv2.imwrite(f"{save_path}/ffflow_{file_name}", ffflowpca)
            cv2.imwrite(f"{save_path}/eventsflow_{file_name}", event_frames_flow)

        dsec_flow[..., 0] = np.clip(dsec_flow[..., 0] * 128 + 2**15, 0, 2**16 - 1)
        dsec_flow[..., 1] = np.clip(dsec_flow[..., 1] * 128 + 2**15, 0, 2**16 - 1)
        dsec_flow = dsec_flow.astype(np.uint16)
        dsec_flow = np.concatenate([dsec_flow, np.ones_like(dsec_flow[..., 0:1])], axis=-1)
        imageio.imwrite(f"{save_path}/{file_name}", dsec_flow, format="PNG-FI")


def main():
    setup_torch(cudnn_benchmark=True)

    model_path = f"outputs/optical_flow/{args.name}/models"
    tsfolder = args.test_forward_optical_flow_timestamps
    data_path = Path(args.datapath)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(confpath, output_dir + "/run.yml")

    logging.basicConfig(filename=f"{output_dir}/exp.log", filemode="w",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Running prediction on: {args.name}")

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
    model.save_configs(output_dir)

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
                                        args.time_ctx, args.time_pred, args.bucket, args.max_numevents_ctx,
                                        randomize_ctx=False, camera="left", dtype="dsec")
        eval_timestamps = np.genfromtxt(eval_timestamp_path, delimiter=",", dtype=np.uint64, skip_header=1)
        eval_timestamps[:, :2] -= h5py.File(event_path, "r")['t_offset'][()].astype(np.uint64)
        test_fixed_time_optical_flow(args, model, event_extractor, eval_timestamps, data_output_path, logger=logger)

if __name__ == '__main__':
    main()
