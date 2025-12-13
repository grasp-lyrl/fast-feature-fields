# Monocular Depth Estimation with F<sup>3</sup>

This directory contains the implementation for training and evaluating monocular depth estimation models using F<sup>3</sup> and other baselines (like voxelgrids and event frames) as backbones.

## Training

To train a depth estimation model using F<sup>3</sup> features:

```bash
python src/f3/tasks/depth/train_monocular_rel.py \
    --conf confs/monocular_depth/dav2b_fullm3ed_pseudo_518x518x20.yml
```

### Key Arguments

- `--conf`: Path to the configuration file (required)
- `--compile`: Enable torch.compile for faster training
- `--amp`: Use automatic mixed precision training
- `--baseline`: Use baseline model without F<sup>3</sup> backbone
- `--retrain_f3`: Allow F<sup>3</sup> backbone to be fine-tuned (default: frozen)
- `--init`: Path to initial weights for the model (to finetune from a checkpoint)
- `--wandb`: Enable Weights & Biases logging
- `--name`: Custom name for the experiment

### Configuration Files

Pre-configured training setups are available in `confs/monocular_depth/`:

- `dav2b_traindsec_finetune_disparity_518x518x20.yml` - Train on DSEC metric depth
- `dav2b_trainm3ed_finetune_disparity_518x518x20.yml` - Train on M3ED metric depth
- `dav2b_fullm3ed_pseudo_518x518x20.yml` - Train on full M3ED with pseudo labels - relative depth
- And more...

## Generating Pseudo Ground Truth

Generate rectified monocular depth maps from RGB/Grayscale images using DepthAnything V2 for any camera in M3ED:

```bash
python src/f3/tasks/depth/generate_depth.py \
    --h5fn /path/to/input_h5 \
    --out_h5fn /path/to/output_h5 \
    --target prophesee \ # camera to warp to
    --side left \ # side of the camera to warp to
    --checkpoint /path/to/depthanythingv2_checkpoint.pth
```

This script:
1. Loads RGB images from the source camera
2. Generates depth predictions using DepthAnything V2
3. Warps the depth maps to the target camera frame using camera calibration
4. Saves the rectified depth maps as HDF5 files

## Model Architecture

The depth estimation pipeline uses:
- **F<sup>3</sup>**: Extracts dense feature representations from events
- **DepthAnything V2**: Processes F<sup>3</sup> features to predict monocular depth

The F<sup>3</sup> backbone is typically frozen during training, with only the DepthAnything V2 decoder being fine-tuned.
