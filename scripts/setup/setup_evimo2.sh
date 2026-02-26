#!/bin/bash
# Script to create symbolic links to the EVIMO2v2 dataset in data/
# and convert sequences to HDF5 for the f3 pipeline.
#
# Usage:
#   ./scripts/setup/setup_evimo2.sh /local/richeek/evimo2v2 data/evimo2
#
# The EVIMO2v2 NPZ format stores events as separate .npy files:
#   dataset_events_xy.npy, dataset_events_t.npy, dataset_events_p.npy
# and masks as:
#   dataset_mask.npz  (with keys mask_0, mask_1, ...)
# with metadata in dataset_info.npz
#
# This script:
#   1. Creates symlinks for each camera/category/split/sequence
#   2. Runs conversion to HDF5 + ms_to_idx generation for the f3 dataloader

eval "$(conda shell.bash hook)"
conda activate f3

BASE_PATH=$1        # Where the EVIMO2v2 dataset is stored, e.g. /local/richeek/evimo2v2
TARGET_PATH=${2:-data/evimo2}   # Where the symbolic links will be created

# Event cameras we care about (skip flea3_7 which is an RGB frame camera at 2080x1552)
CAMERAS=(left_camera samsung_mono right_camera)
# All event-camera categories (imo = standard lighting, imo_ll = low-light)
CATEGORIES=(imo imo_ll)
# Splits
SPLITS=(train eval)

mkdir -p "$TARGET_PATH"

echo "=== Setting up EVIMO2v2 dataset ==="
echo "Source: $BASE_PATH"
echo "Target: $TARGET_PATH"

for camera in "${CAMERAS[@]}"; do
    camera_src="$BASE_PATH/$camera"
    if [ ! -d "$camera_src" ]; then
        echo "WARNING: Camera directory $camera_src not found, skipping (download may not be complete)"
        continue
    fi

    for category in "${CATEGORIES[@]}"; do
        cat_src="$camera_src/$category"
        if [ ! -d "$cat_src" ]; then
            echo "WARNING: Category directory $cat_src not found, skipping"
            continue
        fi

        for split in "${SPLITS[@]}"; do
            split_src="$cat_src/$split"
            if [ ! -d "$split_src" ]; then
                echo "WARNING: Split directory $split_src not found, skipping"
                continue
            fi

            # Create target directory structure
            target_dir="$TARGET_PATH/${camera}/${category}/${split}"
            mkdir -p "$target_dir"

            # Link each sequence folder
            for seq_dir in "$split_src"/*/; do
                seq_name=$(basename "$seq_dir")
                if [ ! -L "$target_dir/$seq_name" ] && [ ! -d "$target_dir/$seq_name" ]; then
                    ln -sf "$seq_dir" "$target_dir/$seq_name"
                    echo "  Linked: $target_dir/$seq_name -> $seq_dir"
                fi
            done
        done
    done
done

echo ""
echo "=== Converting EVIMO2v2 sequences to HDF5 ==="
# Convert all linked sequences to HDF5 format for the f3 dataloader
python3 scripts/convert_evimo2_to_h5.py \
    --base_path "$TARGET_PATH" \
    --cameras ${CAMERAS[@]} \
    --categories ${CATEGORIES[@]} \
    --splits ${SPLITS[@]}

echo ""
echo "=== EVIMO2v2 setup complete ==="
echo "Linked sequences are in: $TARGET_PATH"
echo "HDF5 files are generated alongside each sequence"
