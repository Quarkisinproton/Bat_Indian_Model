"""
Inference script for Bat Species Classification
"""

import argparse
import os
import json
import numpy as np
from PIL import Image
from tensorflow import keras

from src.preprocessing.audio_processor import AudioProcessor
from src.utils.data_loader import BatDataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict bat species from audio or spectrogram')
    
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input file (audio or spectrogram image)')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.h5)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels file (.json)')
    parser.add_argument('--input_type', type=str, default='auto',
                       choices=['auto', 'audio', 'image'],
                       help='Type of input file')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                       help='Image size (height width)')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Number of top predictions to show')
    
    return parser.parse_args()


def detect_input_type(filepath):
    """
    Detect input type based on file extension
    
    Args:
        filepath (str): Path to input file
        
    Returns:
        str: 'audio' or 'image'
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in audio_extensions:
        return 'audio'
    elif ext in image_extensions:
        return 'image'
    else:
        raise ValueError(f"Unknown file type: {ext}")


def load_and_preprocess_image(image_path, img_size):
    """
    Load and preprocess image
    
    Args:
        image_path (str): Path to image
        img_size (tuple): Target image size
        
    Returns:
        np.array: Preprocessed image
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array


def load_and_preprocess_audio(audio_path, img_size):
    """
    Load audio and convert to spectrogram
    
    Args:
        audio_path (str): Path to audio file
        img_size (tuple): Target image size
        
    Returns:
        np.array: Preprocessed spectrogram image
    """
    processor = AudioProcessor()
    img = processor.process_audio_file(audio_path, output_size=img_size)
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array


def predict(args):
    """
    Main prediction function
    
    Args:
        args: Command line arguments
    """
    print("=" * 50)
    print("Bat Species Classification - Prediction")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    
    # Detect input type
    if args.input_type == 'auto':
        input_type = detect_input_type(args.input)
        print(f"\nDetected input type: {input_type}")
    else:
        input_type = args.input_type
    
    # Load and preprocess input
    print(f"\nLoading and preprocessing {input_type}...")
    img_size = tuple(args.img_size)
    
    if input_type == 'audio':
        X = load_and_preprocess_audio(args.input, img_size)
        print(f"Converted audio to spectrogram")
    else:
        X = load_and_preprocess_image(args.input, img_size)
    
    print(f"Input shape: {X.shape}")
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Load labels
    print(f"Loading labels from {args.labels}...")
    with open(args.labels, 'r') as f:
        label_data = json.load(f)
    
    class_names = label_data['class_names']
    
    # Make prediction
    print("\nMaking prediction...")
    predictions = model.predict(X, verbose=0)
    
    # Get top predictions
    top_indices = np.argsort(predictions[0])[-args.top_k:][::-1]
    top_probs = predictions[0][top_indices]
    
    print("\n" + "=" * 50)
    print("Prediction Results:")
    print("=" * 50)
    
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
        species = class_names[idx]
        confidence = prob * 100
        print(f"{i}. {species}: {confidence:.2f}%")
    
    # Return results as dictionary
    results = {
        'input_file': args.input,
        'predictions': [
            {
                'rank': i,
                'species': class_names[idx],
                'confidence': float(prob)
            }
            for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1)
        ]
    }
    
    return results


if __name__ == '__main__':
    args = parse_args()
    results = predict(args)
