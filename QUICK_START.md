# Quick Start Guide

## 5-Minute Setup

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Organize Data
```
data/raw/
├── Species_A/
│   ├── audio1.wav
│   └── audio2.wav
└── Species_B/
    └── audio1.wav
```

### 3. Preprocess
```bash
python preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed
```

### 4. Train
```bash
python train.py \
    --data_dir data/processed \
    --epochs 50 \
    --augment
```

### 5. Predict
```bash
python predict.py \
    --input test.wav \
    --model models/bat_classifier_*.h5 \
    --labels models/bat_classifier_*_labels.json
```

## Command Cheat Sheet

### Training Options
```bash
# Basic CNN
python train.py --data_dir data/processed --model_type cnn

# Transfer Learning (EfficientNet)
python train.py --data_dir data/processed --model_type efficientnet --use_pretrained

# Custom parameters
python train.py --data_dir data/processed --epochs 100 --batch_size 16 --learning_rate 0.0001
```

### Preprocessing Options
```bash
# Custom audio settings
python preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --img_size 224 224 \
    --sample_rate 44100 \
    --duration 5.0
```

### Evaluation
```bash
python evaluate.py \
    --model models/bat_classifier_*.h5 \
    --labels models/bat_classifier_*_labels.json \
    --data_dir data/test
```

## Common Workflows

### Workflow 1: Small Dataset
```bash
# Use augmentation and simple CNN
python train.py \
    --data_dir data/processed \
    --model_type cnn \
    --epochs 50 \
    --augment
```

### Workflow 2: Medium Dataset
```bash
# Use transfer learning with EfficientNet
python train.py \
    --data_dir data/processed \
    --model_type efficientnet \
    --use_pretrained \
    --epochs 50 \
    --learning_rate 0.0001 \
    --augment
```

### Workflow 3: Large Dataset
```bash
# Train from scratch with ResNet
python train.py \
    --data_dir data/processed \
    --model_type resnet \
    --epochs 100 \
    --batch_size 32
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `--batch_size` to 8 or 16 |
| Slow training | Use GPU, reduce image size |
| Poor accuracy | Enable `--augment`, try `--use_pretrained` |
| Overfitting | Increase data, enable augmentation |

## Next Steps

1. Read [README.md](README.md) for detailed documentation
2. Check [USAGE.md](USAGE.md) for advanced examples
3. Run [example_usage.py](example_usage.py) for code samples
4. Monitor training with TensorBoard: `tensorboard --logdir logs/`

## File Outputs

After training, you'll get:
- `*.h5` - Model weights
- `*_labels.json` - Class labels
- `*_config.json` - Training config
- `*_history.json` - Training history

Keep these files together for prediction!
