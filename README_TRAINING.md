# DepthAnything-V2 + ZoeDepth Training Pipeline

This training pipeline combines DepthAnything-V2's pretrained encoder (DINOv2-based) with ZoeDepth's metric depth head to produce metric depth predictions.

## Architecture Overview

1. **DepthAnything-V2 Encoder**: Pretrained DINOv2 ViT encoder that extracts relative depth features
2. **Adapter Layer**: Converts ViT tokens to spatial feature pyramid (1/8, 1/16, 1/32 scales)
3. **Feature Fusion**: Fuses pyramid features into a single 1/8-resolution feature map
4. **ZoeDepth Metric Head**: 
   - Seed bin regressor (initial metric hypothesis)
   - Attractor layers (local bin refinement)
   - Conditional log-binomial distribution (final depth prediction)

## Installation

Ensure you have all dependencies from both DepthAnything-V2 and ZoeDepth:

```bash
pip install torch torchvision
pip install opencv-python pillow numpy scipy
pip install tensorboard tqdm
```

## Model Files

- `da2_zoe_adapter.py`: Adapter module converting DINOv2 tokens to spatial features
- `da2_zoe_model.py`: Combined model architecture
- `train_da2_zoe.py`: Training script with losses, TensorBoard, and checkpointing

## Usage

### Basic Training

```bash
python train_da2_zoe.py \
    --encoder vitl \
    --batch-size 4 \
    --num-epochs 50 \
    --lr 1e-4 \
    --datasets nyu sunrgbd \
    --nyu-csv /path/to/nyu/train.csv \
    --nyu-root /path/to/nyu/data \
    --sunrgbd-json /path/to/sunrgbd/train.json \
    --sunrgbd-root /path/to/sunrgbd/data \
    --save-dir ./checkpoints \
    --log-dir ./logs \
    --train-encoder \
    --validate
```

### Training with Custom Configuration

```bash
python train_da2_zoe.py \
    --encoder vitl \
    --n-bins 64 \
    --bin-centers-type softplus \
    --min-depth 0.1 \
    --max-depth 10.0 \
    --n-attractors 16 8 4 1 \
    --attractor-type inv \
    --batch-size 4 \
    --num-epochs 100 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --lr-step-size 30 \
    --lr-gamma 0.1 \
    --datasets nyu sunrgbd ibims \
    --nyu-csv /path/to/nyu/train.csv \
    --nyu-root /path/to/nyu/data \
    --sunrgbd-json /path/to/sunrgbd/train.json \
    --sunrgbd-root /path/to/sunrgbd/data \
    --ibims-root /path/to/ibims/data \
    --save-dir ./checkpoints \
    --log-dir ./logs \
    --train-encoder \
    --encoder-lr-factor 10 \
    --validate
```

### Resume Training

```bash
python train_da2_zoe.py \
    --resume ./checkpoints/checkpoint_latest.pth \
    --num-epochs 100 \
    # ... other arguments
```

### Loading Pretrained Checkpoint

```bash
python train_da2_zoe.py \
    --pretrained-checkpoint /path/to/pretrained.pth \
    # ... other arguments
```

## Arguments

### Model Configuration
- `--encoder`: Encoder variant (`vits`, `vitb`, `vitl`, `vitg`), default: `vitl`
- `--n-bins`: Number of depth bins, default: `64`
- `--bin-centers-type`: Bin center type (`normed`, `softplus`, `hybrid1`, `hybrid2`), default: `softplus`
- `--min-depth`, `--max-depth`: Depth range, default: `1e-3` to `10.0`
- `--n-attractors`: List of attractor counts at each level, default: `[16, 8, 4, 1]`
- `--attractor-type`: Attractor type (`inv` recommended, or `exp`), default: `inv`
- `--train-encoder`: Whether to train the encoder (use flag to enable)
- `--encoder-lr-factor`: Learning rate factor for encoder, default: `10.0`

### Training Configuration
- `--batch-size`: Batch size, default: `4`
- `--num-epochs`: Number of training epochs, default: `50`
- `--lr`: Learning rate, default: `1e-4`
- `--weight-decay`: Weight decay, default: `1e-4`
- `--lr-step-size`: LR scheduler step size, default: `15`
- `--lr-gamma`: LR scheduler gamma, default: `0.1`
- `--max-grad-norm`: Gradient clipping norm, default: `1.0`

### Loss Configuration
- `--silog-weight`: Weight for SILog loss, default: `1.0`
- `--grad-weight`: Weight for gradient loss, default: `0.5`
- `--metric-silog-weight`: Weight for metric SILog loss, default: `0.5`

### Data Configuration
- `--datasets`: Datasets to use (`nyu`, `sunrgbd`, `ibims`, `da2k`)
- `--target-size`: Target image size, default: `384`
- `--num-workers`: Number of data loader workers, default: `4`

### Dataset Paths
- `--nyu-csv`: Path to NYU dataset CSV file
- `--nyu-root`: Path to NYU dataset root directory
- `--sunrgbd-json`: Path to SUNRGBD JSON annotation file
- `--sunrgbd-root`: Path to SUNRGBD dataset root directory
- `--use-bfx-depth`: Use bfx depth for SUNRGBD (use flag to enable)
- `--ibims-root`: Path to IBIMS dataset root directory
- `--da2k-root`: Path to DA2K dataset root directory
- `--da2k-annotation`: Path to DA2K annotation JSON file

### Checkpointing
- `--save-dir`: Directory to save checkpoints, default: `./checkpoints`
- `--log-dir`: Directory for TensorBoard logs, default: `./logs`
- `--pretrained-checkpoint`: Path to pretrained checkpoint to load
- `--resume`: Path to checkpoint to resume training from

## TensorBoard Visualization

View training progress:

```bash
tensorboard --logdir ./logs
```

TensorBoard logs include:
- Training losses (total, SILog, gradient, metric SILog)
- Validation losses and metrics
- Learning rate schedule
- Depth accuracy metrics (δ1, δ2, δ3, RMSE, AbsRel, etc.)

## Checkpoints

Checkpoints are saved in `--save-dir`:
- `checkpoint_latest.pth`: Latest checkpoint (saved every epoch)
- `checkpoint_epoch_{N}.pth`: Checkpoint for epoch N
- `checkpoint_best.pth`: Best validation checkpoint

## Loss Functions

The training uses a combined loss function:

1. **SILog Loss** (from ZoeDepth): Scale-invariant logarithmic loss
2. **Gradient Loss** (from ZoeDepth): L1 loss on depth gradients
3. **Metric SiLog Loss** (from DepthAnything-V2): Metric-aware scale-invariant loss

Total loss = `silog_weight * SILog + grad_weight * Gradient + metric_silog_weight * MetricSILog`

## Notes

- The encoder is pretrained and should be trained with a lower learning rate (use `--encoder-lr-factor`)
- Use `--train-encoder` flag to enable encoder training (otherwise encoder is frozen)
- Inverse attractor (`--attractor-type inv`) is recommended for better stability
- Batch size should be adjusted based on GPU memory
- The model expects input images to be normalized with ImageNet statistics

## Troubleshooting

1. **Out of memory**: Reduce `--batch-size` or use gradient accumulation
2. **Slow training**: Reduce `--num-workers` or use smaller encoder (`vitb` instead of `vitl`)
3. **NaN losses**: Check data loading, ensure depth values are in valid range
4. **Poor convergence**: Adjust learning rate or loss weights



