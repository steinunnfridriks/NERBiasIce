# train_claude.py

## Purpose
Advanced training and evaluation script for bias detection models. Features class weighting for imbalanced data, early stopping, and comprehensive evaluation metrics. Now supports evaluation-only mode for pre-trained models.

## Usage

### Training a new model
```bash
python train_claude.py
```

### Evaluation only (with pre-trained model)
```bash
python train_claude.py --eval-only --model-dir icebert
```

## Command-line Arguments
- `--eval-only`: Skip training, only evaluate
- `--model-dir`: Directory containing pre-trained model (default: icebert)
- `--output-dir`: Where to save results (default: current directory)

## Features

### Training Enhancements
- **Class weighting**: Automatically balances loss for imbalanced categories
- **Early stopping**: Prevents overfitting (patience=5)
- **Gradient accumulation**: Effective batch size of 32
- **Cosine learning rate schedule**: Better convergence
- **Label smoothing**: Reduces overconfidence
- **Mixed precision (FP16)**: Faster training, less memory

### Evaluation Metrics
- Macro/Micro Precision, Recall, F1
- Per-category metrics for all 14 bias types
- Entity-level evaluation (seqeval)
- Detailed classification reports

### Model Configuration
- Base model: ScandiBERT (vesteinn/ScandiBERT)
- Max sequence length: 128 tokens
- Learning rate: 2e-5
- Epochs: 8 (with early stopping)
- Warmup: 10% of training steps

## Output Files

### Training Mode
- `scandibert_bias_2/`: Model checkpoints
- `training_log.json`: Training history and metrics

### Evaluation Mode
- `test_results.json`: Detailed test set metrics
- `evaluation_report.txt`: Human-readable results

## Data Format
Expects CoNLL-style input files:
- `data/train.txt` (training)
- `data/dev.txt` (validation)
- `data/test.txt` (evaluation)

## Memory Requirements
- GPU: ~8GB VRAM recommended
- RAM: ~16GB recommended
- Disk: ~2GB for model storage

## Performance Monitoring
The script displays:
- Label distribution analysis
- Training progress with loss
- Validation metrics every 200 steps
- Final test set evaluation
- Total training time

## Customization
Key parameters to adjust:
- `per_device_train_batch_size`: Reduce if OOM
- `num_train_epochs`: More for better performance
- `learning_rate`: Lower for fine-tuning
- `early_stopping_patience`: Higher for more training