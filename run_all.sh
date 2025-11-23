#!/bin/bash
set -e

# Activate venv
source venv/bin/activate

echo "Starting CNN Training..."
python src/train.py --model cnn --epochs 10 --batch_size 4

echo "Starting Transformer Training..."
python src/train.py --model transformer --epochs 10 --batch_size 4

echo "Evaluating CNN..."
python src/evaluate.py --checkpoint best_model_cnn.pth --model cnn

echo "Evaluating Transformer..."
python src/evaluate.py --checkpoint best_model_transformer.pth --model transformer

echo "ALL DONE"
