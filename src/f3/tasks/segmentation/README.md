# Semantic Segmentation with F<sup>3</sup>

This directory contains the implementation for training and evaluating semantic segmentation models using F<sup>3</sup> as a backbone.

## Training

To train a segmentation model using F<sup>3</sup> features:

```bash
python src/f3/tasks/segmentation/train.py \
    --conf confs/segmentation/segformer_b3_trainm3ed_1280x720x20.yml
```

### Key Arguments

- `--conf`: Path to the configuration file (required)
- `--compile`: Enable torch.compile for faster training
- `--amp`: Use automatic mixed precision training
- `--wandb`: Enable Weights & Biases logging
- `--name`: Custom name for the experiment

### Training Baseline (Without F<sup>3</sup>)

To train a baseline segmentation model without F<sup>3</sup> backbone (e.g., using voxelgrids or event frames):

```bash
python src/f3/tasks/segmentation/train_baseline.py \
    --conf confs/segmentation/segformer_b3_frames_trainm3ed_1280x720x20.yml
```

### Configuration Files

Pre-configured training setups are available in `confs/segmentation/`:

- `segformer_b3_trainm3ed_1280x720x20.yml` - Train on M3ED dataset using F<sup>3</sup>
- `segformer_b3_frames_trainm3ed_1280x720x20.yml` - Train baseline on M3ED using event frames
- And more...

## Model Architecture

The segmentation pipeline uses:
- **F<sup>3</sup>**: Extracts dense feature representations from events
- **Segformer**: Transformer-based segmentation decoder that processes F<sup>3</sup> features

The F<sup>3</sup> backbone is typically frozen during training, with only the Segformer decoder being fine-tuned.
