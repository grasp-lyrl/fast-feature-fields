"""
A 3D and 2D visualizer for events and trajectories.
"""
import cv2
import torch
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import colormaps as cm


def plot_patched_features(feat, name="patched_features", plot=True):
    """
    Args:
        feat: (W, H, LF) torch.Tensor or np.ndarray
    """
    W, H, LF = feat.shape

    if isinstance(feat, torch.Tensor):
        feat = feat.cpu().numpy()

    if LF == 1:
        pca_result = feat
    elif LF == 2:
        pca_result = np.stack([feat[..., 0], np.zeros_like(feat[..., 0]), feat[..., 1]], axis=-1)
    else:
        feat = feat.reshape(-1, LF)
        # PCA to reduce the dimensionality of the last dimension
        pca = PCA(n_components=min(3, LF))
        pca_result = pca.fit_transform(feat)
        mean, std = pca_result.mean(axis=0), pca_result.std(axis=0)
        pca_result = np.clip(pca_result, mean - std, mean + std)
    pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
    pca_result = (pca_result * 255).astype(np.uint8).reshape(W, H, -1)

    if plot:
        cv2.imwrite(f"{name}_pca.png", pca_result[..., ::-1])

    feat_img = feat.reshape(W, H, LF)
    feat_img = np.linalg.norm(feat_img, axis=-1)
    feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min())
    feat_img = (feat_img * 255).astype(np.uint8)

    if plot:
        cv2.imwrite(f"{name}_levels.png", feat_img)

    return pca_result, feat_img


def display_event_batch(events, name, frame_sizes=(1280, 720)):
    if isinstance(events, torch.Tensor):
        events = events.cpu().numpy()
    if events.max() <= 1:
        events = unnormalize_events(events, frame_sizes)
    img = np.zeros((frame_sizes[1], frame_sizes[0]), dtype=np.uint8)
    img[events[:, 1], events[:, 0]] = 255
    cv2.imwrite(f"{name}.png", img)


def display_event_shifts(events1, events2, name, frame_sizes=(1280, 720)):
    if isinstance(events1, torch.Tensor):
        events1 = events1.cpu().numpy()
    if isinstance(events2, torch.Tensor):
        events2 = events2.cpu().numpy()
    if events1.max() <= 1:
        events1 = unnormalize_events(events1, frame_sizes)
    if events2.max() <= 1:
        events2 = unnormalize_events(events2, frame_sizes)
    img = np.zeros((frame_sizes[1], frame_sizes[0], 3), dtype=np.uint8)
    img[events1[:, 1], events1[:, 0], 0] = 255 # blue
    img[events2[:, 1], events2[:, 0], 2] = 255 # red
    cv2.imwrite(f"{name}.png", img)


def display_predicted_shifts(events1, fullpredframe, name, frame_sizes=(1280, 720)):
    if isinstance(events1, torch.Tensor):
        events1 = events1.cpu().numpy()
    if events1.max() <= 1:
        events1 = unnormalize_events(events1, frame_sizes)
    img = np.zeros((frame_sizes[1], frame_sizes[0], 3), dtype=np.uint8)
    img[events1[:, 1], events1[:, 0], 0] = 255 # blue
    if fullpredframe.ndim == 2:
        img[fullpredframe > 1e-3, 2] = 255 # red
    elif fullpredframe.ndim == 3:
        img[fullpredframe.sum(-1) > 1e-3, 2] = 255 # red
    else:
        raise ValueError("fullpredframe should be 2D or 3D")
    cv2.imwrite(f"{name}.png", img)


def display_predicted_shifts_frames(past_event_frame, fullpredframe, name):
    h, w = past_event_frame.shape[:2]
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., 0] = past_event_frame
    if fullpredframe.ndim == 2:
        img[fullpredframe > 0, 2] = 255
    elif fullpredframe.ndim == 3:
        img[fullpredframe.sum(-1) > 0, 2] = 255
    else:
        raise ValueError("fullpredframe should be 2D or 3D")
    cv2.imwrite(f"{name}.png", img)


def display_event_residuals(events1, events2, name, frame_sizes=(1280, 720)):
    if isinstance(events1, torch.Tensor):
        events1 = events1.cpu().numpy()
    if isinstance(events2, torch.Tensor):
        events2 = events2.cpu().numpy()
    if events1.max() <= 1:
        events1 = unnormalize_events(events1, frame_sizes)
    if events2.max() <= 1:
        events2 = unnormalize_events(events2, frame_sizes)
    img = np.zeros((frame_sizes[1], frame_sizes[0], 3), dtype=np.uint8)
    img[events1[:, 1], events1[:, 0], 0] = 255 # blue
    img[events2[:, 1], events2[:, 0], 2] = 255 # red
    img[(img[..., 0] == 255) & (img[..., 2] == 255)] = 0 # kill the intersection
    cv2.imwrite(f"{name}.png", img)


def smooth_time_weighted_rgb_encoding(img_batch):
    B, W, H, T = img_batch.shape
    time_weights = np.linspace(0, 1, T)**2
    red_channel = np.sum(img_batch * time_weights[None, None, None, :], axis=-1)
    red_channel /= np.max(red_channel, axis=(-2, -1), keepdims=True) + 1e-5
    blue_channel = np.sum(img_batch * (1 - time_weights)[None, None, None, :], axis=-1)
    blue_channel /= np.max(blue_channel, axis=(-2, -1), keepdims=True) + 1e-5
    return (np.stack([blue_channel, np.zeros_like(red_channel), red_channel], axis=-1) * 255).astype(np.uint8)


def unnormalize_events(events, frame_sizes=(1280, 720, 50)):
    if isinstance(events, torch.Tensor):
        return (events[:, :len(frame_sizes)] * torch.tensor(frame_sizes)[None, :].to(events.device)).round().to(torch.int32)
    elif isinstance(events, np.ndarray):
        return (events[:, :len(frame_sizes)] * np.array(frame_sizes)[None, :]).round().astype(np.int32)
    else:
        raise ValueError("events should be either torch.Tensor or np.ndarray")
