import cv2
import torch
import numpy as np

"""
Generates an RGB image where each point corresponds to flow in that direction from the center,
as visualized by flow_viz_tf.
Output: color_wheel_rgb: [1, width, height, 3]
"""
def draw_color_wheel_np(width, height):
    color_wheel_x = np.linspace(-width / 2.,
                                 width / 2.,
                                 width)
    color_wheel_y = np.linspace(-height / 2.,
                                 height / 2.,
                                 height)
    color_wheel_X, color_wheel_Y = np.meshgrid(color_wheel_x, color_wheel_y)
    color_wheel_rgb = flow_viz_np(np.stack([color_wheel_X, color_wheel_Y], axis=-1))
    return color_wheel_rgb


"""
Visualizes optical flow in HSV space using TensorFlow, with orientation as H, magnitude as V.
Returned as RGB.
Input: flow: [width, height, 2]
Output: flow_rgb: [width, height, 3]
"""
def flow_viz_np(flow, norm=False):
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = ang * 180 / np.pi / 2
    hsv[:, :, 1] = 255
    if norm:
        magmean, magstd = np.mean(mag), np.std(mag)
        mag = np.clip(mag, magmean - 2 * magstd, magmean + 2 * magstd)
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_rgb


def eval_flow(pred_flow, gt_flow, valid_mask=None):
    """
    Computes NPE (N-Pixel-Error), EPE (Endpoint Error), and AE (Angular Error) for optical flow.
    
    Args:
        pred_flow: [B, H, W, 2] predicted optical flow
        gt_flow: [B, H, W, 2] ground truth optical flow
        valid_mask: [B, H, W] mask of valid pixels
    """
    if valid_mask is None:
        valid_mask = torch.ones(gt_flow.shape[:-1], dtype=torch.bool)

    valid_pred = pred_flow[valid_mask] # (N, 2)
    valid_gt = gt_flow[valid_mask] # (N, 2)

    # Compute flow error
    flow_error = valid_pred - valid_gt # (N, 2) (u-u_gt, v-v_gt)
    epe = torch.norm(flow_error, dim=-1)  # Endpoint error (EPE) L2 (N,)
    aepe = torch.mean(epe)  # Average endpoint error (AEPE)

    # Compute NPE N=1,2,3
    one_pe = torch.mean((epe > 1).float()) * 100  # Percentage
    two_pe = torch.mean((epe > 2).float()) * 100  # Percentage
    three_pe = torch.mean((epe > 3).float()) * 100  # Percentage

    # Compute Angular Error (AE)
    valid_pred_ = torch.cat([valid_pred, torch.ones(valid_pred.shape[0], 1).cuda()], dim=1) # (N, 3)
    valid_gt_ = torch.cat([valid_gt, torch.ones(valid_gt.shape[0], 1).cuda()], dim=1) # (N, 3)
    ae = torch.acos(torch.clamp(
        torch.sum(valid_pred_ * valid_gt_, dim=1) /
        (torch.norm(valid_pred_, dim=1) * torch.norm(valid_gt_, dim=1)),
        min=-1.0, max=1.0
    )) # Angular error (AE)
    aae = torch.mean(ae) * 180 / np.pi  # Average angular error (AAE)

    return {"1pe": one_pe.item(), "2pe": two_pe.item(), "3pe": three_pe.item(),
            "aepe": aepe.item(), "aae": aae.item()}
