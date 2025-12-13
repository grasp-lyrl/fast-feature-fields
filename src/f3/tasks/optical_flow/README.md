# Optical Flow Estimation with F<sup>3</sup>

This directory contains the implementation for training and evaluating optical flow estimation models using F<sup>3</sup> as a backbone.

## Training

To train an optical flow model using F<sup>3</sup> features:

```bash
python src/f3/tasks/optical_flow/train.py \
    --conf confs/optical_flow/optflow_trainm3ed_20msff_pyr5_28k.yml \
    --compile
```

### Key Arguments

- `--conf`: Path to the configuration file (required)
- `--compile`: Enable torch.compile for faster training
- `--amp`: Use automatic mixed precision training
- `--baseline`: Train baseline model without F<sup>3</sup> backbone (voxelgrids or event frames)
- `--wandb`: Enable Weights & Biases logging
- `--name`: Custom name for the experiment

### Configuration Files

Pre-configured training setups are available in `confs/optical_flow/`:

- `optflow_trainm3ed_20msff_pyr5_28k.yml` - Train on M3ED dataset using F<sup>3</sup>
- And more...

## Generating Ground Truth Flow

Generate optical flow ground truth from LiDAR point clouds for any camera in M3ED:

```bash
python src/f3/tasks/optical_flow/generate_gt.py \
    --events_h5 /path/to/m3ed_events.h5 \
    --depth_h5 /path/to/m3ed_depths.h5 \
    --base_name name_for_output_file.h5
```

### Arguments

- `--events_h5`: Path to M3ED events HDF5 file
- `--depth_h5`: Path to M3ED depth HDF5 file
- `--base_name`: Base name for the output flow HDF5 file
- `--save_movie`: If set, saves a movie visualization of the flow
- `--save_dist`: If set, saves the flow distribution to a png
- `--start_ind`: Starting frame index (optional)
- `--stop_ind`: Ending frame index (optional)

This script:
1. Loads LiDAR point clouds and camera poses from M3ED
2. Computes egomotion from consecutive poses and depth
3. Saves flow maps as HDF5 files with timestamps
4. Optionally generates color-coded flow visualizations

The generated flow ground truth is used in the F<sup>3</sup> framework to evaluate optical flow models on M3ED sequences.

## Model Architecture

The optical flow pipeline uses:
- **F<sup>3</sup>**: Extracts dense feature representations from events
- **FlowHead**: Lightweight decoder that predicts optical flow from F<sup>3</sup> features

The model is trained with a combination of photometric loss and smoothness regularization.
