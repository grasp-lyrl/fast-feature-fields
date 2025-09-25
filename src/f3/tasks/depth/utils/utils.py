import torch
import numpy as np


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / torch.pow(target, 2))

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


def eval_disparity(pred, target):
    assert pred.shape == target.shape

    pred = torch.clamp(pred, min=1e-6)
    target = torch.clamp(target, min=1e-6)

    diff = torch.abs(pred - target)
    diff_log = torch.abs(torch.log(pred) - torch.log(target))

    one_pe = torch.mean((diff > 1).float()) * 100  # Percentage
    two_pe = torch.mean((diff > 2).float()) * 100  # Percentage
    three_pe = torch.mean((diff > 3).float()) * 100  # Percentage

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'1pe': one_pe.item(), '2pe': two_pe.item(), '3pe': three_pe.item(), 'rmse': rmse.item(),
            'rmse_log': rmse_log.item(), 'log10': log10.item(), 'silog': silog.item()}


def get_depth_image(depth, mask, cmap):
    if isinstance(depth, torch.Tensor):
        min_depth = depth[mask].min().item()
        max_depth = depth[mask].max().item()
        depth_colored = torch.clamp(depth, min_depth, max_depth)
        depth_colored = (depth_colored - min_depth) / (max_depth - min_depth)
        depth_colored = depth_colored.squeeze().cpu().numpy()
        depth_colored = (cmap(depth_colored)[:, :, :3] * 255).astype(np.uint8)
        depth_colored[~mask.cpu().numpy()] = 0
    else:
        min_depth = depth[mask].min()
        max_depth = depth[mask].max()
        depth_colored = np.clip(depth, min_depth, max_depth)
        depth_colored = (depth_colored - min_depth) / (max_depth - min_depth)
        depth_colored = (cmap(depth_colored)[:, :, :3] * 255).astype(np.uint8)
        depth_colored[~mask] = 0
    return depth_colored


def get_disparity_image(depth, mask, cmap):
    if isinstance(depth, torch.Tensor):
        min_disp = depth[mask].min().item()
        max_disp = depth[mask].max().item()
        depth_colored = torch.clamp(depth, min_disp, max_disp)
        depth_colored = (depth_colored - min_disp) / (max_disp - min_disp)
        depth_colored = depth_colored.squeeze().cpu().numpy()
        depth_colored = (cmap(depth_colored)[:, :, :3] * 255).astype(np.uint8)
        depth_colored[~mask.cpu().numpy()] = 0
    else:
        min_disp = depth[mask].min()
        max_disp = depth[mask].max()
        depth_colored = np.clip(depth, min_disp, max_disp)
        depth_colored = (depth_colored - min_disp) / (max_disp - min_disp)
        depth_colored = (cmap(depth_colored)[:, :, :3] * 255).astype(np.uint8)
        depth_colored[~mask] = 0
    return depth_colored


def set_best_results(prevbest, newresults):
    for k in prevbest.keys():
        if k in ['d1', 'd2', 'd3']:
            prevbest[k] = max(prevbest[k], newresults[k].item())
        else:
            prevbest[k] = min(prevbest[k], newresults[k].item())


def get_resize_shapes(h: int, w: int, smalleredge: int, multiple: int) -> tuple:
    """
        Assume h < w. We want to make h=518 and calculate w keeping aspect ratio same. ensure w is multiple of 14
        and greater than w/newh * h
    """
    new_h = smalleredge
    new_w = np.ceil(int(w / h * new_h) / multiple) * multiple
    return new_h, int(new_w)
