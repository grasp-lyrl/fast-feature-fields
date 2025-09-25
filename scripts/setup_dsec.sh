# Script to create symbolic links to the DSEC dataset in the prolev/data directory
# Especially useful when the DSEC dataset is stored on a different disk
# + Also generates the 50KHz timestamps for the DSEC dataset

#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate f3

base_path=$1    # Where the DSEC dataset is stored
target_path=$2  # Where the symbolic links will be created'
train=$3        # Whether to create links for the training or testing data, 0 or 1


if [ "$train" == 1 ]; then
    src_dirs=(train_events train_images train_optical_flow)
    srcsub_dirs=(events images flow)
else
    src_dirs=(test_events test_images)
    srcsub_dirs=(events images)
fi

cnt=0
for src_dir in "${src_dirs[@]}"; do
    full_src_dir="$base_path/$src_dir"
    target_subdirs=$(find "$full_src_dir" -maxdepth 1 -mindepth 1 -type d | sed 's/.*\///')

    for target_subdir in $target_subdirs; do
        mkdir -p "$target_path/$target_subdir"
        ln -s "$full_src_dir/$target_subdir/${srcsub_dirs[$cnt]}" "$target_path/$target_subdir"
        if [ "$cnt" == 0 ]; then # Only generate timestamps for the events data
            python3 scripts/generate_ts.py --data_h5 "$target_path/$target_subdir" --dataset "dsec"
        fi
    done
    cnt=$((cnt+1))
done


if [ "$train" == 1 ]; then
    src_dir=train_semantic_segmentation/train
else
    src_dir=test_semantic_segmentation/test
fi
full_src_dir="$base_path/$src_dir"

for target_subdir in $(find "$full_src_dir" -maxdepth 1 -mindepth 1 -type d | sed 's/.*\///'); do
    ln -s "$full_src_dir/$target_subdir" "$target_path/$target_subdir/segmentation"
done


if [ "$train" == 1 ]; then
    src_dir=train_disparity
    full_src_dir="$base_path/$src_dir"

    for target_subdir in $(find "$full_src_dir" -maxdepth 1 -mindepth 1 -type d | sed 's/.*\///'); do
        ln -s "$full_src_dir/$target_subdir" "$target_path/$target_subdir/disparity"
    done
fi
