import os
import cv2
import yaml
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from f3.utils import num_params, unnormalize_events, plot_patched_features, setup_torch
from f3.tasks.segmentation.utils import (EventFFSegformer, EventSegformer,
                                             cityscapes_palette,
                                             get_dataloader_from_args)

import evaluate
mean_iou = evaluate.load("mean_iou")


parser = argparse.ArgumentParser("Train a segmentation model on a dataset of events.")

parser.add_argument("--name", type=str, required=True, help="Name of the run.")
parser.add_argument("--useconfig", type=str, default=None, help="Path to the config file.")
parser.add_argument("--model", type=str, default="best_miou", help="Model to load for evaluation.")
parser.add_argument("--baseline", action="store_true", help="Use the baseline model.")
parser.add_argument("--compile", action="store_true", help="Torch compile both the segmentation model")
parser.add_argument("--amp", action="store_true", help="Use AMP for training.")

args = parser.parse_args()
keys = set(vars(args).keys())

confpath = args.useconfig if args.useconfig else f"outputs/segmentation/{args.name}/args.yml"
with open(confpath, "r") as f:
    conf = yaml.safe_load(f)
for key, value in conf.items():
    if key not in keys:
        setattr(args, key, value)


@torch.no_grad()
def test_fixed_time_segmentation(args, model, val_loader, loss_fn, test_path, logger=None, save_preds=False):
    model.eval()
    test_loss, test_miou, test_acc = 0, 0, 0
    for idx, data in tqdm(enumerate(val_loader), total=len(val_loader)):
       # [(B,N,3) or (B,N,4)], [(B,W,H,2) or (B,W,H,T,2)], [(B,H,W,1)] #! T: max prediction time bins time_pred//bucket
        ff_events, event_counts, _, semantic_labels, src_ofst_res = data

        ff_events = ff_events.cuda()
        event_counts = event_counts.cuda()
        semantic_labels = semantic_labels.cuda().long()

        if not args.polarity[0]: ff_events = ff_events[..., :3]
        
        crop_params = torch.cat([
            src_ofst_res[:, :2],
            src_ofst_res[:, :2] + src_ofst_res[:, 2:]
        ], dim=1).int()
        
        semantic_labels = torch.stack([
            semantic_labels[i, crop_params[i, 0]:crop_params[i, 2], crop_params[i, 1]:crop_params[i, 3]]
            for i in range(semantic_labels.shape[0])
        ])

        seg_logits, ff = model(ff_events, event_counts, crop_params) # (B,N,3) -> (B,C,H,W)
        loss = loss_fn(seg_logits, semantic_labels)

        loss_frames = torch.nn.functional.cross_entropy(seg_logits, semantic_labels,
                                                        reduction='none', ignore_index=255)
        if loss_frames.ndim == 4: loss_frames = loss_frames.sum(-1)
        mincap, maxcap = 0, loss_frames.mean() + 2*loss_frames.std()
        loss_frames = (torch.clamp(loss_frames, mincap, maxcap) - mincap) / (maxcap - mincap)
        loss_frames = (loss_frames * 255).cpu().numpy().astype(np.uint8)
        
        ff = ff.permute(0, 2, 3, 1).cpu().numpy()
        predictions = seg_logits.argmax(1).cpu().numpy()
        semantic_labels = semantic_labels.int().cpu().numpy()
        results = mean_iou.compute(predictions=predictions, references=semantic_labels,
                                   num_labels=args.num_labels, ignore_index=255)

        semantic_labels[semantic_labels == 255] = args.num_labels

        test_loss += loss.item()
        test_miou += results["mean_iou"]
        test_acc += results["overall_accuracy"]

        if save_preds:
            logger.info(f"Saving predictions for batch: {idx}...")

            color_img = cityscapes_palette(args.num_labels)[predictions].astype(np.uint8)
            overlay_image = color_img.copy()
            gt_img = cityscapes_palette(args.num_labels)[semantic_labels].astype(np.uint8)
    
            event_counts = torch.cumsum(torch.cat((torch.zeros(1), event_counts.cpu())), 0).to(torch.uint64)
            event_shift = np.zeros((seg_logits.shape[0], src_ofst_res[0, 2], src_ofst_res[0, 3]), dtype=np.uint8)
            for i in range(seg_logits.shape[0]):
                events = unnormalize_events(ff_events.cpu()[event_counts[i]:event_counts[i+1]].numpy(), args.frame_sizes) -\
                         np.array([src_ofst_res[i, 1], src_ofst_res[i, 0]])
                event_shift[i, events[:, 1], events[:, 0]] = 255 # white -> context events

            overlay_image[event_shift == 255] = 0.5 *overlay_image[event_shift == 255] + 128
            gt_img[event_shift == 255] = 0.5 * gt_img[event_shift == 255] + 128

            for i in range(seg_logits.shape[0]):
                if not args.baseline:
                    ffpca, feat_img = plot_patched_features(ff[i], plot=False)
                    ffpca = ffpca[..., ::-1]
                    overlay_feat = 0.75*color_img[i] + 0.25*ffpca
                    cv2.imwrite(f"{test_path}/predictions/overlayfeat_{idx}_{i}.png", overlay_feat)
                    cv2.imwrite(f"{test_path}/predictions/feat_{idx}_{i}.png", ffpca)

                cv2.imwrite(f"{test_path}/predictions/loss_{idx}_{i}.png", loss_frames[i])
                cv2.imwrite(f"{test_path}/predictions/seg_{idx}_{i}.png", color_img[i])
                cv2.imwrite(f"{test_path}/predictions/overlayev_{idx}_{i}.png", overlay_image[i])
                cv2.imwrite(f"{test_path}/gt/gtseg_{idx}_{i}.png", gt_img[i])
    test_loss /= len(val_loader)
    test_acc /= len(val_loader)
    test_miou /= len(val_loader)

    logger.info("#"*50)
    logger.info(f"Test: Loss: {test_loss}, Acc: {test_acc}, MIoU: {test_miou}")
    logger.info("#"*50)

    return test_loss, test_acc, test_miou


def main():
    setup_torch(cudnn_benchmark=True)
    
    foldername = os.path.basename(confpath).split(".")[0]
    model_path = f"outputs/segmentation/{args.name}/models"
    test_path = f"outputs/segmentation/{args.name}/test/{args.model}/{foldername}"
    
    os.makedirs(test_path, exist_ok=True)
    for dir in ["predictions", "gt"]:
        os.makedirs(f"{test_path}/{dir}", exist_ok=True)
    shutil.copyfile(confpath, test_path + "/run.yml")

    logging.basicConfig(filename=f"{test_path}/exp.log", filemode="a",
                        level=logging.INFO, format="%(asctime)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Running prediction on: {args.name}")

    val_loader = get_dataloader_from_args(args, logger, shuffle=False, train=False)

    logger.info("#"*50)
    if args.baseline:
        model = EventSegformer(args.segformer_config, args.num_labels, args.eventmodel,
                               *args.frame_sizes, args.time_ctx // args.bucket)
    else:
        model = EventFFSegformer(args.eventff["config"], args.segformer_config, num_labels=args.num_labels)
        if not args.compile:
            model.eventff = torch.compile(model.eventff)
    if args.compile:
        model = torch.compile(model)
    model.save_configs(test_path)

    logger.info(f"Feature Field + Segmentation: {model}")
    logger.info(f"Trainable parameters in Segformer: {num_params(model.segformer)}")
    logger.info(f"Total Trainable parameters: {num_params(model)}")
    logger.info("#"*50)

    last_dict = torch.load(f"{model_path}/{args.model}.pth", weights_only=False)
    model.load_state_dict(last_dict["model"], strict=False)
    last_epoch = last_dict["epoch"]
    last_loss = last_dict["loss"]
    last_acc = last_dict["acc"]
    last_miou = last_dict["miou"]
    del last_dict
    torch.cuda.empty_cache()
    logger.info(f"Loaded model from: {args.model}, Epoch: {last_epoch}, Loss: {last_loss}, Acc: {last_acc}, MIoU: {last_miou}")

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    test_fixed_time_segmentation(args, model, val_loader, loss_fn, test_path, logger=logger, save_preds=False)


if __name__ == '__main__':
    main()
