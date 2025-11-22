# Usage Guide

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

Create a directory structure with your bat audio recordings:
```
data/raw/
├── Pipistrellus_pipistrellus/
│   ├── call_001.wav
│   ├── call_002.wav
│   └── ...
├── Rhinolophus_ferrumequinum/
│   ├── call_001.wav
│   └── ...
└── [other species]/
```

### Step 3: Preprocess Audio
```bash
python preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --img_size 128 128
```

### Step 4: Train Model
```bash
python train.py \
    --data_dir data/processed \
    --model_type cnn \
    --epochs 50 \
    --batch_size 32 \
    --augment
```

### Step 5: Make Predictions
```bash
python predict.py \
    --input test_audio.wav \
    --model models/bat_classifier_cnn_[timestamp].h5 \
    --labels models/bat_classifier_cnn_[timestamp]_labels.json
```

## Advanced Usage

### Using Python API

```python
from src.models.bat_classifier import BatSpeciesClassifier
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.data_loader import BatDataLoader

# Initialize components
processor = AudioProcessor()
classifier = BatSpeciesClassifier(num_classes=10, model_type='cnn')

# Build and compile model
classifier.build_model()
classifier.compile_model(learning_rate=0.001)

# Load data
data_loader = BatDataLoader('data/processed')
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()

# Train
history = classifier.model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32
)

# Predict
predictions = classifier.predict(X_test)
```

### Transfer Learning Example

```python
# Use EfficientNet with pretrained weights
classifier = BatSpeciesClassifier(
    num_classes=10,
    model_type='efficientnet',
    use_pretrained=True
)

classifier.build_model()
classifier.compile_model(learning_rate=0.0001)

# Train with frozen base layers
history = classifier.model.fit(X_train, y_train, epochs=20)

# Unfreeze and fine-tune
classifier.unfreeze_base_model(num_layers=50)
classifier.compile_model(learning_rate=0.00001)

# Continue training
history = classifier.model.fit(X_train, y_train, epochs=30)
```

### Audio Augmentation

```python
processor = AudioProcessor()
audio = processor.load_audio('bat_call.wav')

# Apply augmentation
audio_noisy = processor.augment_audio(audio, 'noise')
audio_stretched = processor.augment_audio(audio, 'time_stretch')
audio_shifted = processor.augment_audio(audio, 'pitch_shift')
```

## Configuration

Edit `config/default_config.json` to change default settings:

```json
{
  "model": {
    "type": "cnn",
    "input_shape": [128, 128, 3],
    "num_classes": 10
  },
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

## Tips

1. **Start Small**: Begin with a small subset of data to test the pipeline
2. **Monitor Training**: Use TensorBoard to visualize training progress
3. **Experiment**: Try different model architectures and hyperparameters
4. **Data Quality**: Ensure consistent audio quality across all recordings
5. **Class Balance**: Aim for similar number of samples per species

## Troubleshooting

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--img_size 64 64`

### Poor Performance
- Increase training data
- Enable augmentation: `--augment`
- Try transfer learning: `--model_type efficientnet --use_pretrained`

### Slow Training
- Use GPU acceleration
- Reduce model complexity
- Decrease image resolution
