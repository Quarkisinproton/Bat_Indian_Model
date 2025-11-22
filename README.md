# Bat Species Classification using Deep Learning

A deep learning model for classifying Indian bat species from audio recordings. This project uses Convolutional Neural Networks (CNNs) to analyze audio spectrograms and identify different bat species.

## Features

- **Multiple Model Architectures**: Support for custom CNN, EfficientNet, and ResNet50
- **Audio Processing**: Convert bat call audio recordings to mel spectrograms
- **Data Augmentation**: Built-in augmentation for improved model generalization
- **Transfer Learning**: Option to use pretrained ImageNet weights
- **Comprehensive Training Pipeline**: Includes callbacks for early stopping, learning rate reduction, and checkpointing
- **Easy Inference**: Predict bat species from audio files or spectrograms

## Project Structure

```
Bat_Indian_Model/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── bat_classifier.py      # Model architectures
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── audio_processor.py     # Audio to spectrogram conversion
│   └── utils/
│       ├── __init__.py
│       └── data_loader.py         # Data loading and management
├── config/
│   └── default_config.json        # Default configuration
├── data/
│   ├── raw/                       # Raw audio files (organized by species)
│   ├── processed/                 # Processed spectrograms
│   └── sample/                    # Sample data
├── models/                        # Saved models
├── checkpoints/                   # Training checkpoints
├── logs/                          # TensorBoard logs
├── train.py                       # Training script
├── predict.py                     # Inference script
├── preprocess_data.py            # Data preprocessing script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Quarkisinproton/Bat_Indian_Model.git
cd Bat_Indian_Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Your Data

Organize your audio files in the following structure:
```
data/raw/
├── species_1/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── species_2/
│   ├── audio1.wav
│   └── ...
└── ...
```

### 2. Preprocess Audio Data

Convert audio files to spectrograms:
```bash
python preprocess_data.py \
    --input_dir data/raw \
    --output_dir data/processed \
    --img_size 128 128 \
    --sample_rate 44100 \
    --n_mels 128 \
    --duration 5.0
```

### 3. Train the Model

Train a model on your processed data:

**Basic training with custom CNN:**
```bash
python train.py \
    --data_dir data/processed \
    --model_type cnn \
    --epochs 50 \
    --batch_size 32 \
    --augment
```

**Transfer learning with EfficientNet:**
```bash
python train.py \
    --data_dir data/processed \
    --model_type efficientnet \
    --use_pretrained \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --augment
```

**Advanced training with custom parameters:**
```bash
python train.py \
    --data_dir data/processed \
    --model_type resnet \
    --use_pretrained \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --optimizer adam \
    --test_size 0.2 \
    --val_size 0.1 \
    --augment \
    --output_dir models \
    --checkpoint_dir checkpoints
```

### 4. Make Predictions

Predict bat species from audio files or spectrograms:

**From audio file:**
```bash
python predict.py \
    --input path/to/audio.wav \
    --model models/bat_classifier_cnn_20231122_120000.h5 \
    --labels models/bat_classifier_cnn_20231122_120000_labels.json \
    --top_k 3
```

**From spectrogram image:**
```bash
python predict.py \
    --input path/to/spectrogram.png \
    --model models/bat_classifier_cnn_20231122_120000.h5 \
    --labels models/bat_classifier_cnn_20231122_120000_labels.json \
    --input_type image \
    --top_k 3
```

## Model Architectures

### 1. Custom CNN
A custom convolutional neural network with:
- 4 convolutional blocks with batch normalization and dropout
- Progressive feature map increase (32 → 64 → 128 → 256)
- Dense layers with dropout for classification
- Best for: Small to medium datasets, fast training

### 2. EfficientNet-B0
Transfer learning using EfficientNet:
- Pretrained on ImageNet
- Efficient scaling of depth, width, and resolution
- Fine-tuning capability
- Best for: Limited data, high accuracy requirements

### 3. ResNet50
Transfer learning using ResNet50:
- Pretrained on ImageNet
- Deep residual learning framework
- Proven performance on image classification
- Best for: Large datasets, complex patterns

## Training Parameters

Key parameters that can be adjusted:

- `--model_type`: Model architecture (cnn, efficientnet, resnet)
- `--use_pretrained`: Use pretrained weights for transfer learning
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate for optimizer
- `--optimizer`: Optimizer type (adam, sgd, rmsprop)
- `--augment`: Enable data augmentation
- `--test_size`: Proportion of data for testing (default: 0.2)
- `--val_size`: Proportion of training data for validation (default: 0.1)

## Audio Processing

The audio processor converts bat call recordings to mel spectrograms with the following parameters:

- **Sample Rate**: 44100 Hz (default)
- **Mel Bands**: 128 (default)
- **FFT Window**: 2048 samples
- **Hop Length**: 512 samples
- **Duration**: 5 seconds (default)

## Data Augmentation

When training with `--augment`, the following augmentations are applied:
- Random rotation (±15 degrees)
- Width and height shifts (±10%)
- Horizontal flipping
- Zoom (±10%)

## Monitoring Training

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir logs/
```

## Model Outputs

After training, the following files are saved:
- `*.h5`: Trained model weights
- `*_labels.json`: Label encoder and class names
- `*_config.json`: Training configuration and test results
- `*_history.json`: Training history (loss, accuracy per epoch)

## Evaluation Metrics

The model reports the following metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **AUC**: Area Under the ROC Curve

## Example Output

```
Prediction Results:
==================================================
1. Pipistrellus pipistrellus: 87.32%
2. Rhinolophus ferrumequinum: 8.45%
3. Myotis daubentonii: 3.21%
```

## Requirements

- Python 3.7+
- TensorFlow 2.8+
- Keras 2.8+
- NumPy
- Librosa (for audio processing)
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Pillow

See `requirements.txt` for complete list.

## Tips for Best Results

1. **Data Quality**: Ensure audio recordings are clear and properly labeled
2. **Data Balance**: Try to have similar number of samples per species
3. **Preprocessing**: Experiment with different audio parameters (n_mels, duration)
4. **Model Selection**: Start with custom CNN, use transfer learning if needed
5. **Augmentation**: Enable augmentation for small datasets
6. **Early Stopping**: Training automatically stops if no improvement
7. **Learning Rate**: Reduce if model isn't converging

## Common Issues

**Issue**: Low accuracy on test set
- Solution: Increase dataset size, enable augmentation, try transfer learning

**Issue**: Model overfitting
- Solution: Increase dropout, enable augmentation, reduce model complexity

**Issue**: Training too slow
- Solution: Reduce batch size, use smaller model, use GPU acceleration

**Issue**: Out of memory
- Solution: Reduce batch size, reduce image size, use smaller model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Citation

If you use this model in your research, please cite:
```
@software{bat_indian_model,
  title={Bat Species Classification using Deep Learning},
  author={Quarkisinproton},
  year={2023},
  url={https://github.com/Quarkisinproton/Bat_Indian_Model}
}
```

## Acknowledgments

- TensorFlow and Keras teams for the deep learning framework
- Librosa developers for audio processing tools
- The bat research community for domain knowledge

## Contact

For questions or issues, please open an issue on GitHub.