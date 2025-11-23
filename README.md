# Bat Species Identification and Call Counting

This project implements deep learning models (CNN and Transformer) to identify bat species and count their calls in audio recordings.

## Project Structure

```
bat_project/
├── data/
│   ├── annotations.json  # Annotations file (provided)
│   └── ...               # WAV files (please upload here)
├── src/
│   ├── dataset.py        # Data loading and processing
│   ├── models.py         # CNN and Transformer model definitions
│   ├── train.py          # Training script
│   └── evaluate.py       # Evaluation script
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    -   Ensure `annotations.json` is in `data/`.
    -   Upload your 50 WAV files to `data/` (or any subdirectory within `data/`). The code will automatically find them by filename.

## Usage

### Training

To train the model (default is CNN):

```bash
cd src
python train.py --epochs 20 --batch_size 4 --model cnn
```

To train the Transformer model:

```bash
python train.py --epochs 20 --batch_size 4 --model transformer
```

**Arguments**:
-   `--model`: `cnn` or `transformer`
-   `--epochs`: Number of training epochs (default: 20)
-   `--batch_size`: Batch size (default: 4)
-   `--lr`: Learning rate (default: 1e-4)
-   `--count_weight`: Weight for the count regression loss (default: 1.0)

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint best_model_cnn.pth --model cnn
```

**Arguments**:
-   `--checkpoint`: Path to the saved model weights (e.g., `best_model_cnn.pth`)
-   `--model`: `cnn` or `transformer`

## Models

-   **BatCNN**: A standard Convolutional Neural Network operating on Mel Spectrograms.
-   **BatTransformer**: A Vision Transformer (ViT) adapted for audio spectrogram patches.

Both models output:
1.  **Species Classification**: Logits for 2 species.
2.  **Call Count**: A scalar value representing the number of calls.
