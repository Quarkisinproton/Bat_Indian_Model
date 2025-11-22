# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Bat Species Classification                │
│                      Deep Learning System                    │
└─────────────────────────────────────────────────────────────┘

┌───────────────┐
│  Audio Input  │  (.wav, .mp3, .flac, .ogg)
└───────┬───────┘
        │
        ▼
┌───────────────────────┐
│  Audio Processor      │
│  - Load Audio         │
│  - Normalize          │
│  - Pad/Trim           │
│  - Mel Spectrogram    │
│  - Augmentation       │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  Spectrogram Image    │  (128x128x3 RGB)
└───────┬───────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│            Model Architecture             │
│  ┌─────────────────────────────────────┐ │
│  │  Option 1: Custom CNN               │ │
│  │  - 4 Conv Blocks                    │ │
│  │  - BatchNorm + Dropout              │ │
│  │  - Dense Layers                     │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │  Option 2: EfficientNet-B0          │ │
│  │  - Pretrained on ImageNet           │ │
│  │  - Fine-tuning capable              │ │
│  └─────────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐ │
│  │  Option 3: ResNet50                 │ │
│  │  - Deep Residual Network            │ │
│  │  - Transfer Learning                │ │
│  └─────────────────────────────────────┘ │
└───────┬───────────────────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Species Predictions  │
│  - Class Probabilities│
│  - Top-K Results      │
│  - Confidence Scores  │
└───────────────────────┘
```

## Data Flow

### Training Pipeline

```
Raw Audio Files
      ↓
[Audio Preprocessing]
      ↓
Spectrogram Images
      ↓
[Data Loading & Splitting]
      ↓
Train / Validation / Test Sets
      ↓
[Data Augmentation]
      ↓
[Model Training]
      ↓
Trained Model + Metadata
```

### Inference Pipeline

```
New Audio File
      ↓
[Audio Preprocessing]
      ↓
Spectrogram Image
      ↓
[Load Trained Model]
      ↓
[Prediction]
      ↓
Species Classification
```

## Custom CNN Architecture

```
Input: (128, 128, 3)
│
├─ Conv2D(32, 3x3) → BatchNorm → ReLU
├─ Conv2D(32, 3x3) → BatchNorm → ReLU
├─ MaxPool2D(2x2) → Dropout(0.25)
│
├─ Conv2D(64, 3x3) → BatchNorm → ReLU
├─ Conv2D(64, 3x3) → BatchNorm → ReLU
├─ MaxPool2D(2x2) → Dropout(0.25)
│
├─ Conv2D(128, 3x3) → BatchNorm → ReLU
├─ Conv2D(128, 3x3) → BatchNorm → ReLU
├─ MaxPool2D(2x2) → Dropout(0.25)
│
├─ Conv2D(256, 3x3) → BatchNorm → ReLU
├─ Conv2D(256, 3x3) → BatchNorm → ReLU
├─ MaxPool2D(2x2) → Dropout(0.25)
│
├─ Flatten()
├─ Dense(512) → BatchNorm → ReLU → Dropout(0.5)
├─ Dense(256) → BatchNorm → ReLU → Dropout(0.5)
│
└─ Dense(num_classes) → Softmax
```

## Module Structure

### 1. Audio Processing Module (`src/preprocessing/`)
- **AudioProcessor**: Main class for audio operations
  - `load_audio()`: Load and resample audio
  - `preprocess_audio()`: Normalize and pad/trim
  - `audio_to_melspectrogram()`: Convert to mel spectrogram
  - `audio_to_mfcc()`: Extract MFCC features
  - `augment_audio()`: Apply data augmentation
  - `batch_process_directory()`: Batch processing

### 2. Model Module (`src/models/`)
- **BatSpeciesClassifier**: Main model class
  - `build_cnn_model()`: Custom CNN architecture
  - `build_efficientnet_model()`: EfficientNet with transfer learning
  - `build_resnet_model()`: ResNet50 with transfer learning
  - `compile_model()`: Configure optimizer and loss
  - `predict()`: Make predictions
  - `save_model()` / `load_model()`: Model persistence

### 3. Utilities Module (`src/utils/`)
- **BatDataLoader**: Data management
  - `load_data_from_directory()`: Load organized dataset
  - `prepare_data()`: Split into train/val/test
  - `decode_predictions()`: Convert predictions to labels
  - `save_label_encoder()` / `load_label_encoder()`: Label management

- **Data Generators**: Training utilities
  - `create_data_generators()`: Create augmented data generators

## Training Components

### Callbacks
1. **ModelCheckpoint**: Save best model during training
2. **EarlyStopping**: Stop when validation loss stops improving
3. **ReduceLROnPlateau**: Reduce learning rate on plateau
4. **TensorBoard**: Log training metrics for visualization

### Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **AUC**: Area under ROC curve

### Data Augmentation
- Rotation (±15°)
- Width/Height shifts (±10%)
- Horizontal flipping
- Zoom (±10%)

## Audio Processing Details

### Mel Spectrogram Parameters
- **Sample Rate**: 44100 Hz (default)
- **N_FFT**: 2048 (FFT window size)
- **Hop Length**: 512 (stride between windows)
- **N_Mels**: 128 (number of mel frequency bins)
- **Duration**: 5 seconds (default segment length)

### Frequency Range
- Bat echolocation calls: typically 20-120 kHz
- Model adapts to any frequency content in recordings

## Model Comparison

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|--------|----------|----------|
| Custom CNN | ~2M | Fast | Good | Small datasets, quick experiments |
| EfficientNet-B0 | ~5M | Medium | Better | Medium datasets, balanced performance |
| ResNet50 | ~25M | Slow | Best | Large datasets, maximum accuracy |

## Performance Considerations

### Memory Usage
- **Custom CNN**: ~500MB GPU memory
- **EfficientNet**: ~1GB GPU memory
- **ResNet50**: ~2GB GPU memory

### Training Time (per epoch, 1000 samples)
- **Custom CNN**: ~30 seconds (GPU), ~5 minutes (CPU)
- **EfficientNet**: ~60 seconds (GPU), ~10 minutes (CPU)
- **ResNet50**: ~90 seconds (GPU), ~15 minutes (CPU)

## Extensibility

The architecture is designed to be extensible:

1. **Add New Models**: Implement new `build_*_model()` methods
2. **Custom Augmentation**: Extend `augment_audio()` in AudioProcessor
3. **Feature Extraction**: Add methods like `audio_to_mfcc()` for new features
4. **Custom Metrics**: Add to `compile_model()` metrics list

## Best Practices

1. **Start Simple**: Begin with Custom CNN
2. **Transfer Learning**: Use when data is limited (<1000 samples/class)
3. **Augmentation**: Always enable for small datasets
4. **Batch Size**: Adjust based on GPU memory
5. **Learning Rate**: Use 0.001 for training from scratch, 0.0001 for fine-tuning
6. **Early Stopping**: Let training stop automatically
7. **Validation**: Use separate validation set for hyperparameter tuning
