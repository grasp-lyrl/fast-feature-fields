import random
import string
import numpy as np

import torch
from torch.utils.data.dataloader import default_collate

from .utils_viz import unnormalize_events


def generate_alphanumeric(length=7):
  """Generates a random alphanumeric string of the specified length."""

  characters = string.ascii_letters + string.digits
  return ''.join(random.choice(characters) for i in range(length))


def crop_params(src_ofst_res: torch.Tensor, crop_size: torch.Tensor):
    """
        src_ofst_res: [B, 4] -> [src_ofst_x, src_ofst_y, src_res_x, src_res_y]
        crop_size: [2] -> [crop_size_x, crop_size_y]
        
        We want atleast 1/4th of the crop to have the source image in it.
    """
    limits = torch.stack([
        torch.max(src_ofst_res[:, :2] - crop_size//2, torch.zeros(2, dtype=torch.int32)), # 120 - 512//2
        torch.min(src_ofst_res[:, :2] + src_ofst_res[:, 2:] - crop_size//2,   # 120 + 480 - 512//2
                  2*src_ofst_res[:, :2] + src_ofst_res[:, 2:] - crop_size)    # 120*2 + 480 - 256
    ], dim=1)
    ranges = limits[:, 1] - limits[:, 0] # [B, 2]
    offsets = limits[:,0] # [B, 2]
    cparams = (torch.rand(src_ofst_res.shape[0], 2) * ranges + offsets).int() # [B, 2]
    return torch.cat([cparams, cparams + crop_size], dim=1) # [B, 4]


@torch.compile
def batch_cropper(images, cparams):
    """
    Crop a batch of images using the provided crop parameters.

    Parameters:
        images: Batch of images [B, C, H, W]
        cparams: Crop parameters [B, 4]

    Returns:
        Cropped images [B, C, H, W]
    """
    if cparams is None:
        return images
    return torch.stack([
        images[i, :, cparams[i, 0]:cparams[i, 2], cparams[i, 1]:cparams[i, 3]]
        for i in range(images.shape[0])
    ])


def calculate_receptive_field(layers):
    """
    Calculate the receptive field for a sequence of layers.

    Parameters:
        layers: List of tuples [(kernel_size1, stride1, dilation1), (kernel_size2, stride2, dilation2), ...]

    Returns:
        Receptive field size at the final layer.
    """
    receptive_field = 1
    product = 1
    for kernel_size, stride, dilation in layers:
        product *= stride
        receptive_field += (kernel_size - 1) * dilation * product
    return receptive_field


@torch.compile
def ev_to_grid(events: torch.Tensor, counts: torch.Tensor, w: int, h: int, t: int) -> torch.Tensor:
    """
        Converts given events and event counts per batch to a voxel grid
        
        events: (N,3) N = N1+N2+...+NB tensor
        counts: (B,)      N1,N2,...,NB tensor
        
        Returns: (B, W, H, T) tensor
    """
    if events.max() <= 1.0: events = unnormalize_events(events, (w, h, t))
    B = counts.shape[0]
    pred_event_grid = torch.zeros(B, w, h, t, device=events.device, dtype=torch.float32)
    c = torch.cumsum(torch.cat((torch.zeros(1), counts.cpu())), 0).to(torch.uint64)
    for i in range(B):
        pred_event_grid[i, events[c[i]:c[i+1], 0], events[c[i]:c[i+1], 1], events[c[i]:c[i+1], 2]] = 1
    return pred_event_grid


@torch.compile
def ev_to_frames(events, counts, w, h):
    """
        Converts given events and event counts per batch to a voxel grid
        
        events: (N,3) N = N1+N2+...+NB ndarray or tensor
        counts: (B,)      N1,N2,...,NB ndarray or tensor
        
        Returns: (B, W, H) tensor
    """
    if events.max() <= 1.0: events = unnormalize_events(events, (w, h))
    B = counts.shape[0]

    if isinstance(events, torch.Tensor):
        event_frames = torch.zeros(B, w, h, dtype=torch.uint8).to(events.device)
        c = torch.cumsum(torch.cat((torch.zeros(1).to(counts.device), counts)), 0).to(torch.int32)
    elif isinstance(events, np.ndarray):
        event_frames = np.zeros((B, w, h), dtype=np.uint8)
        c = np.cumsum(np.concatenate((np.zeros(1), counts))).astype(np.int32)

    for i in range(B):
        event_frames[i, events[c[i]:c[i+1], 0], events[c[i]:c[i+1], 1]] = 255
    return event_frames


def collate_fn_general(batch, concat_ids, collate_ids):
    """
    Collate function to handle a batch of tuples where the first element is
    a variable-sized tensor to be concatenated, and the rest are handled separately.
    concatids always come first, then collate ids
    Args:
        batch (list of tuple): A list of tuples where each tuple contains:
                            (torch.Tensor of shape (ni, 3/4), *other_elements).
        concat_ids (list of int): List of indices to concatenate.
        collate_ids (list of int): List of indices to default_collate.
    Returns:
        List of torch.Tensor: Concatenated tensors followed by default_collated tensors.
    """
    concat = lambda idx: torch.cat([item[idx] for item in batch], dim=0)
    collate = lambda idx: default_collate([item[idx] for item in batch])
    return [concat(idx) for idx in concat_ids] + [collate(idx) for idx in collate_ids]


def tp_tn_fp_fn(predictions: torch.Tensor, pred_event_grid: torch.Tensor):
    """
    Calculate true positives, true negatives, false positives, and false negatives.

    Parameters:
        predictions: Predictions from the model [B, W, H, T]
        pred_event_grid: Predicted event grid [B, W, H, T]

    Returns:
        Tuple of true positives, true negatives, false positives, and false negatives.
    """
    tp = ((predictions > 0.5) & (pred_event_grid == 1)).sum()
    tn = ((predictions < 0.5) & (pred_event_grid == 0)).sum()
    fp = ((predictions > 0.5) & (pred_event_grid == 0)).sum()
    fn = ((predictions < 0.5) & (pred_event_grid == 1)).sum()
    return tp, tn, fp, fn


def acc_f1(tp, tn, fp, fn):
    """
    Calculate accuracy and F1 score.

    Parameters:
        tp: True positives
        tn: True negatives
        fp: False positives
        fn: False negatives
    """
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    return acc, f1


def log_dict(logger, dict):
    msg = ""
    for k, v in dict.items():
        if isinstance(v, float):
            msg += f"{k}: {v:.4f}, "
        elif isinstance(v, int):
            msg += f"{k}: {v}, "
        elif isinstance(v, torch.Tensor):
            msg += f"{k}: {v.item()}, "
        else:
            msg += f"{k}: {v}, "
    logger.info(msg)
