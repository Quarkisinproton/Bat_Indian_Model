import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

class BatDataset(Dataset):
    def __init__(self, annotations_file, data_dir, target_sample_rate=384000, n_mels=128, fixed_duration=None):
        """
        Args:
            annotations_file (str): Path to the JSON annotations file.
            data_dir (str): Directory containing the WAV files.
            target_sample_rate (int): Sample rate to resample audio to.
            n_mels (int): Number of mel filterbanks.
            fixed_duration (float): If set, pad/crop audio to this duration (in seconds).
        """
        self.data_dir = Path(data_dir)
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.fixed_duration = fixed_duration
        
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
            
        self.tags_map = {tag['id']: tag['value'] for tag in self.annotations['data']['tags']}
        self.samples = self._parse_annotations()
        
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def _parse_annotations(self):
        data = self.annotations['data']
        recordings = {rec['uuid']: rec for rec in data['recordings']}
        sound_events = data['sound_events']
        
        # Aggregate events per recording
        recording_events = {}
        for event in sound_events:
            rec_uuid = event['recording']
            if rec_uuid not in recording_events:
                recording_events[rec_uuid] = []
            recording_events[rec_uuid].append(event)
            
        samples = []
        for rec_uuid, rec_info in recordings.items():
            events = recording_events.get(rec_uuid, [])
            call_count = len(events)
            
            # Determine species from events (assuming single species per file for now)
            # If no events, we might need to infer from path or skip. 
            # For this task, we assume we have annotations for the 50 files.
            species_id = -1
            if events:
                # Look for species tag in events
                for event in events:
                    if event['tags']:
                        species_id = event['tags'][0] # Assuming first tag is species
                        break
            
            # If species_id is still -1, try to guess from path (fallback)
            if species_id == -1:
                if "Pip ceylonicus" in rec_info['path']:
                    species_id = 0 # Based on JSON tags: 0 is Sco. heathii/ Pip. ceylonicus
                elif "Pip tenuis" in rec_info['path']: # Guessing path structure for other species
                    species_id = 1
            
            # Find local file path
            filename = Path(rec_info['path']).name
            local_path = self._find_file(filename)
            
            if local_path:
                samples.append({
                    'path': str(local_path),
                    'species_id': species_id,
                    'call_count': call_count,
                    'orig_duration': rec_info['duration']
                })
            else:
                print(f"Warning: File {filename} not found in {self.data_dir}")
                
        return samples

    def _find_file(self, filename):
        # Search in data_dir recursively
        for path in self.data_dir.rglob(filename):
            return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        waveform, sample_rate = torchaudio.load(sample['path'])
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Handle duration (pad or crop)
        if self.fixed_duration:
            target_len = int(self.fixed_duration * self.target_sample_rate)
            current_len = waveform.shape[1]
            if current_len < target_len:
                padding = target_len - current_len
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif current_len > target_len:
                waveform = waveform[:, :target_len]
        
        # Generate Spectrogram
        spec = self.mel_spectrogram(waveform)
        spec = self.amplitude_to_db(spec)
        
        return {
            'spectrogram': spec,
            'species_label': torch.tensor(sample['species_id'], dtype=torch.long),
            'call_count': torch.tensor(sample['call_count'], dtype=torch.float),
            'path': sample['path']
        }
