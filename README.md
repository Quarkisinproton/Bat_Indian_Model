# Bat Species Identification and Call Counting

This project implements a Deep Learning framework to identify bat species and count their calls using audio recordings.

## Project Structure
- `data/`: Directory for storing WAV files and annotations.
- `src/`: Source code.
    - `dataset.py`: Data loading and processing.
    - `models.py`: CNN and Transformer model definitions.
    - `train.py`: Training script.
    - `dummy_data_gen.py`: Script to generate dummy data for testing.
- `requirements.txt`: List of dependencies.

## Setup

1.  **Install Dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Data Preparation**:
    - Place your 50 WAV files in `data/`.
    - Place your JSON annotation file at `data/annotations.json`.
    - Ensure the JSON format matches the expected structure (list of dicts with `filename`, `species`, `count`).

    *Testing with Dummy Data*:
    If you don't have the data yet, you can generate dummy data:
    ```bash
    python src/dummy_data_gen.py
    ```

## Training

To train the CNN model:
```bash
python src/train.py --model cnn --epochs 20
```

To train the Transformer model:
```bash
python src/train.py --model transformer --epochs 20
```

## Models
- **CNN**: A standard convolutional network with global pooling.
- **Transformer**: A Vision Transformer (ViT) adapted for spectrograms.

Both models output:
1.  **Species Class**: Classification logits.
2.  **Call Count**: Regression output.
