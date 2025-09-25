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
import torch.nn as nn

from f3.tasks.segmentation.utils import FFSegformer, cityscapes_palette, get_dataloaders_from_args


parser = argparse.ArgumentParser("Train a segmentation model on a dataset of events.")

parser.add_argument("--useconfig", type=str, required=True, help="Path to the config file.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the segmentation model")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = args.useconfig
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


@torch.no_grad()
def test_image_segmentation(model, val_loader, test_path, logger=None, save_preds=False):
    model.eval()
    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
       # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        image = data[-1].cuda()
        # crop the image to the required size 720, 1280
        image = image[:, :720, :1280, :]
        if image.shape[-1] == 1:
            image = image.expand(-1, -1, -1, 3)
        image = image.permute(0, 3, 1, 2).float() / 255.0
        seg_logits = model(image) # (B,N,3) -> (B,C,H,W)

        predictions = seg_logits.argmax(1).cpu().numpy()
        predictions[predictions == 255] = 19

        if save_preds:
            logger.info(f"Saving predictions for batch: {idx}...")

            color_img = cityscapes_palette()[predictions].astype(np.uint8)
            overlay = (0.5 * color_img + 0.5 * image.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
            for i in range(seg_logits.shape[0]):
                cv2.imwrite(f"{test_path}/predictions/overlay_{idx}_{i}.png", overlay[i])
                cv2.imwrite(f"{test_path}/predictions/seg_{idx}_{i}.png", color_img[i])

def main():
    torch.manual_seed(403)
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True # turn on for faster training if we are using the fixed event mode
    torch.set_float32_matmul_precision('high')
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    torch._dynamo.config.capture_scalar_outputs = True
    
    assert args.train_batch % args.mini_batch == 0, "train_batch should be divisible by mini_batch"
    assert "eventff" in args, "EventFF Model path should be in config"

    foldername = os.path.basename(confpath).split(".")[0]
    test_path = f"outputs/segmentation/{foldername}"
    
    os.makedirs(test_path, exist_ok=True)
    for dir in ["predictions"]:
        os.makedirs(f"{test_path}/{dir}", exist_ok=True)
    shutil.copyfile(confpath, test_path + "/run.yml")

    logging.basicConfig(filename=f"{test_path}/exp.log", filemode="a",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Running prediction on: {args.name}")

    _, val_loader = get_dataloaders_from_args(args, logger)

    logger.info("#"*50)
    model = FFSegformer(args.segformer_config, input_channels=3)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.save_configs(test_path)
    logger.info(f"Model: {args.segformer_config}")
    logger.info("#"*50)

    test_image_segmentation(model, val_loader, test_path, logger=logger, save_preds=True)


if __name__ == '__main__':
    main()
