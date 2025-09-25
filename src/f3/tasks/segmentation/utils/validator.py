import cv2
import torch
import wandb
import numpy as np
from tqdm import tqdm

from f3.utils import unnormalize_events
from f3.tasks.segmentation.utils import cityscapes_palette

import evaluate
mean_iou = evaluate.load("mean_iou")


@torch.no_grad()
def validate_fixed_time_segmentation(args, model, val_loader, loss_fn, epoch,
                                     logger=None, save_preds=False):
    #! Technically just supports batch size 1 if multiple resolutions are used
    model.eval()
    val_loss, val_miou, val_acc = 0, 0, 0
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

        seg_logits = model(ff_events, event_counts, crop_params)[0] # (B,N,3) -> (B,C,H,W)
        loss = loss_fn(seg_logits, semantic_labels)

        predictions = seg_logits.argmax(1).cpu().numpy()
        semantic_labels = semantic_labels.int().cpu().numpy()
        results = mean_iou.compute(predictions=predictions, references=semantic_labels,
                                   num_labels=args.num_labels, ignore_index=255)

        val_loss += loss.item()
        val_miou += results["mean_iou"]
        val_acc += results["overall_accuracy"]

        if idx % 20 == 0 and save_preds:
            logger.info(f"Saving predictions for epoch: {epoch} and batch: {idx}...")

            predictions[predictions == 255] = args.num_labels
            semantic_labels[semantic_labels == 255] = args.num_labels
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
                cv2.imwrite(f"outputs/segmentation/{args.name}/predictions/seg_{epoch}_{idx}_{i}.png", color_img[i])
                cv2.imwrite(f"outputs/segmentation/{args.name}/predictions/overlay_{epoch}_{idx}_{i}.png", overlay_image[i])
                cv2.imwrite(f"outputs/segmentation/{args.name}/training_events/gtseg_{epoch}_{idx}_{i}.png", gt_img[i])
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    val_miou /= len(val_loader)

    logger.info("#"*50)
    logger.info(f"Validation: Epoch: {epoch}, Loss: {val_loss}, Acc: {val_acc}, MIoU: {val_miou}")
    logger.info("#"*50)

    if args.wandb:
        wandb.log({"val_acc": val_acc, "val_loss": val_loss, "val_miou": val_miou, "epoch": epoch})

    return val_loss, val_acc, val_miou
