import os
import json
import numpy as np
import scipy.io.wavfile as wav
import random

def generate_dummy_data(data_dir="data", num_files=50, duration=5.0, sample_rate=22050):
    os.makedirs(data_dir, exist_ok=True)
    
    species_list = ["Pipistrellus", "Myotis"]
    annotations = []
    
    print(f"Generating {num_files} dummy wav files in {data_dir}...")
    
    for i in range(num_files):
        filename = f"bat_call_{i}.wav"
        filepath = os.path.join(data_dir, filename)
        
        # Generate random noise/sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Random frequency between 20kHz and 80kHz (simulating bat range, though sampled at 22k for simplicity in this dummy)
        # Note: 22k sample rate can only represent up to 11kHz. 
        # For real bat calls we'd need higher SR, but for dummy code verification 22k is fine.
        freq = random.uniform(200, 5000) 
        audio = 0.5 * np.sin(2 * np.pi * freq * t)
        audio += 0.1 * np.random.normal(size=len(t)) # Add noise
        
        wav.write(filepath, sample_rate, audio.astype(np.float32))
        
        # Create annotation
        species = random.choice(species_list)
        count = random.randint(1, 10)
        
        annotations.append({
            "filename": filename,
            "species": species,
            "count": count
        })
        
    json_path = os.path.join(data_dir, "annotations.json")
    with open(json_path, 'w') as f:
        json.dump(annotations, f, indent=4)
        
    print(f"Created annotations at {json_path}")

if __name__ == "__main__":
    generate_dummy_data(data_dir="/home/gb/.gemini/antigravity/scratch/bat_project/data")
