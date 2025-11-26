"""
PyTorch Inference Script for Bat Species Classification
"""

import torch
import torch.nn as nn
import librosa
import numpy as np
import argparse
from pathlib import Path
import importlib.util
import sys

def load_model(model_path, model_type='cnn', n_mels=128, device='cuda'):
    """Load trained PyTorch model"""
    # Import models using importlib workaround
    spec = importlib.util.spec_from_file_location("models_pytorch", Path(__file__).parent / "src" / "models.py")
    models_pytorch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models_pytorch)
    
    if model_type == 'cnn':
        model = models_pytorch.BatCNN(n_mels=n_mels, n_classes=2)
    elif model_type == 'transformer':
        model = models_pytorch.BatTransformer(n_mels=n_mels, n_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    
    return model

def predict_species(audio_path, sample_rate=44100, n_mels=128, duration=3.5):
    """Predict species from audio file - returns dict with results"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load audio
    waveform, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Trim/pad to fixed duration
    target_samples = int(sample_rate * duration)
    if len(waveform) > target_samples:
        waveform = waveform[:target_samples]
    else:
        waveform = np.pad(waveform, (0, target_samples - len(waveform)), mode='constant')
    
    # Generate mel-spectrogram
    spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalize
    spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-9)
    
    # Convert to tensor (batch_size=1, channels=1, height=n_mels, width=time)
    spec_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Predict - model now only returns species_logits (not count_pred)
    with torch.no_grad():
        species_logits = model(spec_tensor)
    
    # Get prediction
    species_probs = torch.softmax(species_logits, dim=1)
    species_id = torch.argmax(species_probs, dim=1).item()
    confidence = species_probs[0, species_id].item() * 100
    
    species_names = ["Pip ceylonicus", "Pip tenuis"]
    
    return {
        'species': species_names[species_id],
        'confidence': confidence,
        'species_id': species_id,
        'probs': species_probs.cpu().numpy()[0]
    }

