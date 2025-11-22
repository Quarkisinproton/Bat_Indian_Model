import os
import json
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
from collections import Counter

class BatDataset(Dataset):
    def __init__(self, data_dir, annotation_file, target_sample_rate=22050, duration=5.0, n_mels=128):
        """
        Args:
            data_dir (str): Directory with wav files.
            annotation_file (str): Path to the JSON annotation file.
            target_sample_rate (int): Sample rate to resample to.
            duration (float): Fixed duration in seconds to pad/crop audio.
            n_mels (int): Number of Mel bands for spectrogram.
        """
        self.data_dir = data_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = int(target_sample_rate * duration)
        self.n_mels = n_mels
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Pre-process annotations into a list of dicts
        self.data = self._parse_annotations(self.annotations)
        
        # Create class mapping
        # Filter out 'Unknown' if necessary or handle it
        species_list = sorted(list(set(d['species'] for d in self.data)))
        self.species_to_idx = {species: i for i, species in enumerate(species_list)}
        self.idx_to_species = {i: s for s, i in self.species_to_idx.items()}
        
        print(f"Loaded {len(self.data)} files.")
        print(f"Classes: {self.species_to_idx}")

    def _parse_annotations(self, annotations):
        """
        Normalize annotation format.
        Handles the specific 'Whombat' / 'annotation_project' format.
        """
        data = []
        
        # Check for the specific format provided (nested under 'data')
        if isinstance(annotations, dict) and 'data' in annotations and 'recordings' in annotations['data']:
            print("Detected complex JSON format. Parsing...")
            data_block = annotations['data']
            
            # 1. Parse Tags (ID -> Species Name)
            # Example: [{"id":0,"key":"Species","value":" Sco. heathii/ Pip. ceylonicus"}, ...]
            tags_map = {}
            for t in data_block.get('tags', []):
                if t.get('key') == 'Species':
                    tags_map[t['id']] = t['value'].strip()
            
            # 2. Parse Recordings (UUID -> Filename)
            rec_map = {}
            for rec in data_block.get('recordings', []):
                # Extract basename from path (e.g. "C:\\Users\\...\\Pcy14.wav" -> "Pcy14.wav")
                path = rec.get('path', '')
                # Handle Windows paths if present
                filename = os.path.basename(path.replace('\\', '/'))
                rec_map[rec['uuid']] = {
                    'filename': filename,
                    'count': 0,
                    'species_ids': []
                }
            
            # 3. Parse Sound Events (Link to Recording + Tags)
            for event in data_block.get('sound_events', []):
                rec_uuid = event.get('recording')
                if rec_uuid in rec_map:
                    rec_map[rec_uuid]['count'] += 1
                    # Collect tags to determine species later
                    if event.get('tags'):
                        rec_map[rec_uuid]['species_ids'].extend(event['tags'])
            
            # 4. Flatten to list
            for uuid, info in rec_map.items():
                # Determine species
                species = "Unknown"
                if info['species_ids']:
                    # Use the most frequent tag as the label for the file
                    # (Assuming files are mostly single-species)
                    most_common_id = Counter(info['species_ids']).most_common(1)[0][0]
                    species = tags_map.get(most_common_id, "Unknown")
                
                # Only add if we have a valid species (or keep Unknown if you want)
                # For now, we keep everything
                data.append({
                    'filename': info['filename'],
                    'species': species,
                    'count': int(info['count'])
                })
                
            return data

        # Fallback to simple list/dict format (for dummy data compatibility)
        print("Using simple format parser.")
        if isinstance(annotations, list):
            for entry in annotations:
                filename = entry.get('filename') or entry.get('file')
                species = entry.get('species') or entry.get('label')
                count = entry.get('count') or entry.get('call_count') or 0
                if filename and species:
                    data.append({'filename': filename, 'species': species, 'count': int(count)})
        elif isinstance(annotations, dict):
            for filename, details in annotations.items():
                species = details.get('species') or details.get('label')
                count = details.get('count') or details.get('call_count') or 0
                if species:
                    data.append({'filename': filename, 'species': species, 'count': int(count)})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = os.path.join(self.data_dir, item['filename'])
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            # print(f"Error loading {audio_path}: {e}")
            # Return a dummy tensor in case of error (e.g. file missing)
            return torch.zeros((1, self.n_mels, 216)), torch.tensor(0), torch.tensor(0.0)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)

        # Mix down to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Pad or crop to fixed duration
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        elif waveform.shape[1] < self.num_samples:
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Convert to Mel Spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_mels=self.n_mels,
            n_fft=1024,
            hop_length=512
        )(waveform)
        
        # Convert to dB scale
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # Labels
        species_idx = self.species_to_idx[item['species']]
        call_count = torch.tensor(item['count'], dtype=torch.float32)

        return mel_spectrogram, torch.tensor(species_idx, dtype=torch.long), call_count
