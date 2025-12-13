"""
This script has been kep independent of the f3 package
to make it easier to test the exported pt2 files on orin.
This way one does not need to install the f3 package on orin,
which has many dependencies and is a bit cumbersome to install.

Author: Richeek Das
"""

import cv2
import h5py
import time
import argparse
import numpy as np
from sklearn.decomposition import PCA

import torch
import torch._inductor
import torch.nn.functional as F

torch.set_float32_matmul_precision('high')


def plot_events(events: torch.Tensor, img_width: int, img_height: int):
    """Plot events onto an image and save to disk."""
    img = torch.zeros((img_height, img_width, 3), device=events.device)
    coords_x = (events[:, 0] * img_width).round().int()
    coords_y = (events[:, 1] * img_height).round().int()
    polarities = events[:, 3]
    img[coords_y, coords_x, 0] += (polarities == 1).float()
    img[coords_y, coords_x, 2] += (polarities == 0).float()
    img = torch.clamp(img, 0, 1)
    if img.device.type == 'cuda':
        img = img.cpu()
    img = (img.numpy() * 255).astype("uint8")
    cv2.imwrite("events_plot.png", img)
    return img


def plot_patched_features(feat):
    """
    Args:
        feat: (W, H, LF) torch.Tensor or np.ndarray
    """
    W, H, LF = feat.shape

    if isinstance(feat, torch.Tensor):
        feat = feat.cpu().numpy()

    feat = feat.reshape(-1, LF)
    # PCA to reduce the dimensionality of the last dimension
    pca = PCA(n_components=min(3, LF))
    pca_result = pca.fit_transform(feat)
    mean, std = pca_result.mean(axis=0), pca_result.std(axis=0)
    pca_result = np.clip(pca_result, mean - std, mean + std)
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    pca_result = (pca_result * 255).astype(np.uint8).reshape(W, H, -1)
    cv2.imwrite("f3_feat_pca.png", pca_result)


def plot_depth(depth):
    """
    Plot depth map with proper normalization and save to disk.
    
    Args:
        depth: torch.Tensor or np.ndarray, depth map to visualize
        output_filename: str, filename to save the visualization
    """
    # Postprocessing, implement this part in your chosen language:
    # cmap = cv2.applyColorMap(np.arange(0, 256, dtype=np.uint8), cv2.COLORMAP_JET)
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    
    if depth.ndim == 3:
        depth = depth[0]

    mask = ~np.isnan(depth)
    min_depth = depth[mask].min()
    max_depth = depth[mask].max()
    depth_colored = np.clip(depth, min_depth, max_depth)
    depth_colored = (depth_colored - min_depth) / (max_depth - min_depth)
    depth_colored[~mask] = 0
    depth_colored = (depth_colored * 255).astype(np.uint8)
    cv2.imwrite("depth_colored.png", depth_colored)


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


def plot_flow(flow, event_img=None):
    """
    Plot optical flow visualization and save to disk.
    
    Args:
        flow: torch.Tensor or np.ndarray of shape (B, 2, H, W) or (2, H, W) or (H, W, 2)
    """
    if isinstance(flow, torch.Tensor):
        if flow.ndim == 4:  # (B, 2, H, W)
            flow = flow[0]  # Take first batch
        flow = flow.cpu().numpy()

    flow_rgb = flow_viz_np(flow.transpose(1, 2, 0) if flow.shape[0] == 2 else flow)

    if event_img is not None:
        overlay_image = flow_rgb.copy()
        mask = (event_img == 255).astype(np.uint8).max(axis=-1)
        # interpolate mask to match overlay_image size if needed
        if overlay_image.shape[0] != mask.shape[0] or overlay_image.shape[1] != mask.shape[1]:
            mask = cv2.resize(mask, (overlay_image.shape[1], overlay_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay_image *= mask[..., None]
        cv2.imwrite("flow_viz_overlay.png", overlay_image)

    cv2.imwrite("flow_viz.png", flow_rgb)


def load_events_from_h5(h5_file_path, time_ms,
                        img_width=1280, img_height=720, time_window_ms=20):
    """
    Load events from M3ED H5 file and return normalized events tensor.
    
    Args:
        h5_file_path: str, path to the H5 file
        start_time_ms: int, start time in milliseconds
        n_events: int, maximum number of events to load
        img_width: int, image width for normalization (default: 1280)
        img_height: int, image height for normalization (default: 720)
        time_window_ms: int, time window in milliseconds (default: 20)
    
    Returns:
        torch.Tensor: normalized events tensor of shape (N, 4) on CUDA
    """
    hdf5_file = h5py.File(h5_file_path, 'r')
    events_x = hdf5_file['prophesee/left/x']
    events_y = hdf5_file['prophesee/left/y']
    events_t = hdf5_file['prophesee/left/t']
    events_p = hdf5_file['prophesee/left/p']
    ms_to_idx = hdf5_file['prophesee/left/ms_map_idx']

    norm_factor = torch.tensor([
        img_width, img_height, time_window_ms, 1
    ], dtype=torch.float32)

    ei = ms_to_idx[time_ms]
    si = ms_to_idx[time_ms - time_window_ms]

    events_input = torch.zeros((ei - si, 4), dtype=torch.float32)
    events_input[:, 0] = torch.from_numpy(events_x[si:ei])
    events_input[:, 1] = torch.from_numpy(events_y[si:ei])
    events_input[:, 2] = time_ms - torch.from_numpy(events_t[si:ei]) // 1000 - 1
    events_input[:, 3] = torch.from_numpy(events_p[si:ei])

    events_input = events_input / norm_factor

    events_input = events_input.cuda()

    return events_input


def count_parameters(model):
    """Count total parameters in a model"""
    return sum(p.numel() for p in model.parameters())


def compile_engine_and_infer(f3_pt2_path, dav2_pt2_path, flowhead_pt2_path, h5_file_path, time_ms, 
                             runs, warmup_runs, dav2_height, dav2_width, flow_height, flow_width):
    # Load events from H5 file
    ctx = load_events_from_h5(h5_file_path, time_ms)
    event_img = plot_events(ctx, 1280, 720)

    aoti_f3_compiled = torch._inductor.aoti_load_package(f3_pt2_path)
    if dav2_pt2_path is not None:
        aoti_dav2_compiled = torch._inductor.aoti_load_package(dav2_pt2_path)
    if flowhead_pt2_path is not None:
        aoti_flowhead_compiled = torch._inductor.aoti_load_package(flowhead_pt2_path)

    # Warm-up runs
    for _ in range(warmup_runs):
        f3_feat = aoti_f3_compiled(ctx).permute(0, 3, 2, 1)
        if dav2_pt2_path is not None:
            f3_feat_ds = F.interpolate(
                f3_feat,
                size=(dav2_height, dav2_width),
                mode='bilinear',
                align_corners=True
            )
            depth = aoti_dav2_compiled(f3_feat_ds.float())
        if flowhead_pt2_path is not None:
            f3_feat_flow = F.interpolate(
                f3_feat,
                size=(flow_height, flow_width),
                mode='bilinear',
                align_corners=True
            )
            flow = aoti_flowhead_compiled(f3_feat_flow.float())

    if dav2_pt2_path is not None:
        print(f"Depth output shape: {depth.shape}")
    if flowhead_pt2_path is not None:
        print(f"Flow output shape: {flow.shape}")

    plot_patched_features(f3_feat[0].permute(1, 2, 0))

    if dav2_pt2_path is not None:
        plot_depth(depth.float())
    if flowhead_pt2_path is not None:
        plot_flow(flow.float(), event_img=event_img)

    # Timed runs
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(runs):
        f3_feat = aoti_f3_compiled(ctx).permute(0, 3, 2, 1)
        if dav2_pt2_path is not None:
            f3_feat_ds = F.interpolate(
                f3_feat,
                size=(dav2_height, dav2_width),
                mode='bilinear',
                align_corners=True,
            )
            depth = aoti_dav2_compiled(f3_feat_ds.float())
        if flowhead_pt2_path is not None:
            f3_feat_flow = F.interpolate(
                f3_feat,
                size=(flow_height, flow_width),
                mode='bilinear',
                align_corners=True,
            )
            flow = aoti_flowhead_compiled(f3_feat_flow.float())
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    print(f"Average inference time over {runs} runs: {total_time / runs * 1000:.2f} ms")

    torch._dynamo.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AOTI inference test')
    parser.add_argument('--f3_pt2_path', type=str, required=True,
                        help='Path to the F3 pt2 file')
    parser.add_argument('--dav2_pt2_path', type=str, default=None,
                        help='Path to the DAV2 pt2 file')
    parser.add_argument('--flowhead_pt2_path', type=str, default=None,
                        help='Path to the FlowHead pt2 file')
    parser.add_argument('--h5_file', type=str, default='car_urban_day_penno_small_loop_data.h5',
                        help='Path to the M3ED H5 dataset file (default: car_urban_day_penno_small_loop_data.h5)')
    parser.add_argument('--time_ms', type=int, default=30000,
                        help='Time in milliseconds for event extraction (default: 30000)')
    parser.add_argument('--runs', type=int, default=200,
                        help='Number of timed inference runs (default: 200)')
    parser.add_argument('--warmup_runs', type=int, default=20,
                        help='Number of warmup runs before timing (default: 20)')
    parser.add_argument('--dav2_height', type=int, default=238,
                        help='Target height for DAV2 input (default: 238)')
    parser.add_argument('--dav2_width', type=int, default=308,
                        help='Target width for DAV2 input (default: 308)')
    parser.add_argument('--flow_height', type=int, default=238,
                        help='Target height for FlowHead input (default: 238)')
    parser.add_argument('--flow_width', type=int, default=308,
                        help='Target width for FlowHead input (default: 308)')

    args = parser.parse_args()

    print("⚙️ Configuration:")
    print(f"    H5 file: {args.h5_file}")
    print(f"    Start time: {args.time_ms} ms")
    print(f"    F3 PT2 path: {args.f3_pt2_path}")
    print(f"    DAV2 PT2 path: {args.dav2_pt2_path if args.dav2_pt2_path else 'None'}")
    print(f"    FlowHead PT2 path: {args.flowhead_pt2_path if args.flowhead_pt2_path else 'None'}")
    print(f"    DAV2 input shape: {args.dav2_height}x{args.dav2_width}")
    print(f"    FlowHead input shape: {args.flow_height}x{args.flow_width}")
    print(f"    Warmup runs: {args.warmup_runs}")
    print(f"    Timed runs: {args.runs}")
    print()

    compile_engine_and_infer(
        args.f3_pt2_path,
        args.dav2_pt2_path,
        args.flowhead_pt2_path,
        args.h5_file,
        args.time_ms,
        args.runs,
        args.warmup_runs,
        args.dav2_height,
        args.dav2_width,
        args.flow_height,
        args.flow_width
    )
