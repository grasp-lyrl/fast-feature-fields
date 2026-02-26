#!/usr/bin/env python3
"""
Convert EVIMO2v2 NPZ sequences to HDF5 format for the f3 dataloader.

Each EVIMO2v2 sequence folder contains:
  - dataset_events_xy.npy  (N, 2) uint16
  - dataset_events_t.npy   (N,) float32  (seconds)
  - dataset_events_p.npy   (N,) uint8
  - dataset_mask.npz        mask_0, mask_1, ... (H, W) uint16
  - dataset_info.npz        index, discretization, K, D, meta

This script creates an HDF5 file per sequence with:
  - events/x, events/y, events/t (in microseconds, uint64), events/p
  - ms_to_idx: millisecond-to-event-index mapping for the dataloader
  - masks/mask_<id>: each mask frame
  - mask_timestamps: timestamps of each mask frame (seconds)
  - mask_object_ids: list of unique object IDs across masks
  - info/K, info/D, info/meta (preserved)

Usage:
    python scripts/convert_evimo2_to_h5.py --base_path data/evimo2 \
        --cameras left_camera samsung_mono --categories imo --splits train eval
"""

import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path


def get_sequence_folders(base_path, cameras, categories, splits):
    """Find all sequence folders under the given base path."""
    folders = []
    for camera in cameras:
        for category in categories:
            for split in splits:
                search_dir = Path(base_path) / camera / category / split
                if not search_dir.exists():
                    print(f"  Skipping {search_dir} (not found)")
                    continue
                for seq_dir in sorted(search_dir.iterdir()):
                    # Follow symlinks
                    resolved = seq_dir.resolve() if seq_dir.is_symlink() else seq_dir
                    if resolved.is_dir():
                        folders.append((str(seq_dir), camera, category, split, seq_dir.name))
    return folders


def convert_sequence(seq_path, output_h5_path, force=False):
    """Convert a single EVIMO2v2 sequence to HDF5."""
    seq_path = Path(seq_path).resolve()

    if Path(output_h5_path).exists() and not force:
        print(f"  Skipping {output_h5_path} (already exists)")
        return True

    # Check required files exist
    events_xy_path = seq_path / "dataset_events_xy.npy"
    events_t_path = seq_path / "dataset_events_t.npy"
    events_p_path = seq_path / "dataset_events_p.npy"
    info_path = seq_path / "dataset_info.npz"
    mask_path = seq_path / "dataset_mask.npz"

    for p in [events_xy_path, events_t_path, events_p_path, info_path]:
        if not p.exists():
            print(f"  ERROR: Missing {p}")
            return False

    # Load events (memory-mapped for efficiency)
    events_xy = np.load(str(events_xy_path), mmap_mode="r")
    events_t = np.load(str(events_t_path), mmap_mode="r")
    events_p = np.load(str(events_p_path), mmap_mode="r")

    num_events = events_t.shape[0]
    if num_events == 0:
        print(f"  WARNING: No events in {seq_path}, skipping")
        return False

    # Load info
    info = np.load(str(info_path), allow_pickle=True)
    meta = info["meta"].item()
    K = info["K"]
    D = info["D"]

    # Convert timestamps from seconds to microseconds
    # events_t is float32 in seconds
    print(f"  Converting {num_events:,} events...")
    chunk_size = 10_000_000
    events_t_us = np.empty(num_events, dtype=np.int64)
    for i in range(0, num_events, chunk_size):
        end = min(i + chunk_size, num_events)
        events_t_us[i:end] = (events_t[i:end] * 1_000_000).astype(np.int64)

    # Build ms_to_idx: maps millisecond index to event index
    # ms_to_idx[i] = first event index where event_time_us >= i * 1000
    t_start_us = int(events_t_us[0])
    t_end_us = int(events_t_us[-1])
    t_start_ms = t_start_us // 1000
    t_end_ms = t_end_us // 1000 + 2  # +2 for safety margin

    print(f"  Building ms_to_idx ({t_end_ms + 1:,} entries, absolute 0-{t_end_ms} ms)...")
    # Build ms_to_idx with absolute indexing: ms_to_idx[i] = first event index where t >= i*1000 us
    # Starting from ms=0 ensures the dataloader can use t_us // 1000 directly as the array index.
    ms_timestamps = np.arange(0, t_end_ms + 1) * 1000  # in us

    # Use searchsorted for efficient mapping
    ms_to_idx = np.searchsorted(events_t_us, ms_timestamps, side="left").astype(np.uint64)

    # Load masks
    has_masks = mask_path.exists()
    mask_timestamps = []
    mask_frames = []
    mask_object_ids = set()

    if has_masks:
        masks_npz = np.load(str(mask_path), allow_pickle=True)
        mask_keys = sorted(masks_npz.keys(), key=lambda k: int(k.split("_")[-1]))
        print(f"  Loading {len(mask_keys)} mask frames...")

        # Get timestamps from meta['frames']
        frames_meta = meta.get("frames", [])
        for i, frame_info in enumerate(frames_meta):
            if i < len(mask_keys):
                mask_timestamps.append(frame_info["ts"])
                mask_data = masks_npz[mask_keys[i]]
                mask_frames.append(mask_data)
                # Object IDs are mask_value // 1000 (0 = background)
                unique_vals = np.unique(mask_data)
                for v in unique_vals:
                    oid = int(v) // 1000
                    if oid > 0:
                        mask_object_ids.add(oid)

    # Write HDF5
    print(f"  Writing {output_h5_path}...")
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)

    with h5py.File(output_h5_path, "w") as h5f:
        # Events
        h5f.create_dataset("events/x", data=events_xy[:, 0].astype(np.uint16))
        h5f.create_dataset("events/y", data=events_xy[:, 1].astype(np.uint16))
        h5f.create_dataset("events/t", data=events_t_us)
        h5f.create_dataset("events/p", data=events_p[:].astype(np.uint8))

        # Millisecond to index mapping (for the f3 BaseExtractor)
        h5f.create_dataset("ms_to_idx", data=ms_to_idx)

        # Store the ms offset so we know what ms index 0 corresponds to
        h5f.attrs["t_start_ms"] = t_start_ms
        h5f.attrs["t_start_us"] = t_start_us
        h5f.attrs["t_end_us"] = t_end_us
        h5f.attrs["num_events"] = num_events

        # Camera info
        h5f.create_dataset("info/K", data=K)
        h5f.create_dataset("info/D", data=D)

        # Masks
        if has_masks and len(mask_frames) > 0:
            h5f.attrs["has_masks"] = True
            h5f.create_dataset("mask_timestamps", data=np.array(mask_timestamps, dtype=np.float64))
            # Store masks individually (compressed)
            for i, mask in enumerate(mask_frames):
                h5f.create_dataset(f"masks/mask_{i}", data=mask, compression="gzip", compression_opts=4)
            h5f.attrs["num_mask_frames"] = len(mask_frames)
            h5f.attrs["mask_object_ids"] = sorted(list(mask_object_ids))
            h5f.attrs["mask_resolution"] = mask_frames[0].shape  # (H, W)
        else:
            h5f.attrs["has_masks"] = False

        # Store inner meta for resolution info
        inner_meta = meta.get("meta", {})
        if inner_meta:
            h5f.attrs["res_x"] = inner_meta.get("res_x", 0)
            h5f.attrs["res_y"] = inner_meta.get("res_y", 0)

    print(f"  Done: {output_h5_path} ({os.path.getsize(output_h5_path) / 1e6:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert EVIMO2v2 NPZ sequences to HDF5")
    parser.add_argument("--base_path", type=str, required=True, help="Base path containing camera/category/split/sequence folders")
    parser.add_argument("--cameras", nargs="+", default=["left_camera", "samsung_mono", "right_camera"])
    parser.add_argument("--categories", nargs="+", default=["imo", "imo_ll"])
    parser.add_argument("--splits", nargs="+", default=["train", "eval"])
    parser.add_argument("--force", action="store_true", help="Overwrite existing HDF5 files")
    args = parser.parse_args()

    folders = get_sequence_folders(args.base_path, args.cameras, args.categories, args.splits)
    print(f"Found {len(folders)} sequences to convert")

    success, fail = 0, 0
    for seq_path, camera, category, split, seq_name in tqdm(folders, desc="Converting"):
        resolved_path = Path(seq_path).resolve()
        h5_path = os.path.join(str(resolved_path), "events.h5")
        print(f"\n[{camera}/{category}/{split}/{seq_name}]")
        if convert_sequence(seq_path, h5_path, force=args.force):
            success += 1
        else:
            fail += 1

    print(f"\n=== Conversion complete: {success} succeeded, {fail} failed ===")


if __name__ == "__main__":
    main()
