"""
Audio preprocessing module for bat call analysis
Converts audio files to spectrograms for deep learning
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import os


class AudioProcessor:
    """
    Process audio files and convert them to spectrograms for model input
    """
    
    def __init__(self, sample_rate=44100, n_mels=128, n_fft=2048, 
                 hop_length=512, duration=5.0):
        """
        Initialize audio processor
        
        Args:
            sample_rate (int): Sample rate for audio processing
            n_mels (int): Number of mel bands
            n_fft (int): FFT window size
            hop_length (int): Hop length for STFT
            duration (float): Duration of audio segments in seconds
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.duration = duration
        self.max_length = int(sample_rate * duration)
    
    def load_audio(self, audio_path):
        """
        Load audio file
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            np.array: Audio signal
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        return audio
    
    def preprocess_audio(self, audio):
        """
        Preprocess audio: normalize and pad/trim
        
        Args:
            audio (np.array): Audio signal
            
        Returns:
            np.array: Preprocessed audio
        """
        # Pad or trim to fixed length
        if len(audio) < self.max_length:
            audio = np.pad(audio, (0, self.max_length - len(audio)))
        else:
            audio = audio[:self.max_length]
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        return audio
    
    def audio_to_melspectrogram(self, audio):
        """
        Convert audio to mel spectrogram
        
        Args:
            audio (np.array): Audio signal
            
        Returns:
            np.array: Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def audio_to_mfcc(self, audio, n_mfcc=13):
        """
        Extract MFCC features from audio
        
        Args:
            audio (np.array): Audio signal
            n_mfcc (int): Number of MFCC coefficients
            
        Returns:
            np.array: MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return mfcc
    
    def spectrogram_to_image(self, spectrogram, size=(128, 128)):
        """
        Convert spectrogram to RGB image
        
        Args:
            spectrogram (np.array): Spectrogram data
            size (tuple): Output image size
            
        Returns:
            np.array: RGB image array
        """
        # Normalize to 0-255 range
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        # Handle constant spectrograms to avoid division by zero
        if spec_max == spec_min:
            spec_norm = np.zeros_like(spectrogram)
        else:
            spec_norm = (spectrogram - spec_min) / (spec_max - spec_min)
        
        spec_norm = (spec_norm * 255).astype(np.uint8)
        
        # Convert to PIL Image and resize
        img = Image.fromarray(spec_norm)
        img = img.resize(size, Image.LANCZOS)
        
        # Convert to RGB
        img_rgb = Image.new('RGB', size)
        img_rgb.paste(img)
        
        return np.array(img_rgb)
    
    def save_spectrogram_image(self, spectrogram, output_path, cmap='viridis'):
        """
        Save spectrogram as image file
        
        Args:
            spectrogram (np.array): Spectrogram data
            output_path (str): Path to save image
            cmap (str): Colormap for visualization
        """
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            spectrogram,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            cmap=cmap
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def process_audio_file(self, audio_path, output_size=(128, 128)):
        """
        Complete pipeline: load audio, convert to spectrogram, return as image
        
        Args:
            audio_path (str): Path to audio file
            output_size (tuple): Output image size
            
        Returns:
            np.array: RGB image array of spectrogram
        """
        # Load and preprocess audio
        audio = self.load_audio(audio_path)
        audio = self.preprocess_audio(audio)
        
        # Convert to mel spectrogram
        mel_spec = self.audio_to_melspectrogram(audio)
        
        # Convert to image
        img = self.spectrogram_to_image(mel_spec, size=output_size)
        
        return img
    
    def batch_process_directory(self, input_dir, output_dir, output_size=(128, 128),
                                audio_extensions=['.wav', '.mp3', '.flac', '.ogg']):
        """
        Process all audio files in a directory
        
        Args:
            input_dir (str): Input directory with audio files
            output_dir (str): Output directory for spectrograms
            output_size (tuple): Output image size
            audio_extensions (list): List of audio file extensions to process
            
        Returns:
            list: List of processed file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        processed_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    input_path = os.path.join(root, file)
                    
                    # Create corresponding output path
                    rel_path = os.path.relpath(root, input_dir)
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = os.path.splitext(file)[0] + '.png'
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    try:
                        # Process audio
                        img = self.process_audio_file(input_path, output_size)
                        
                        # Save as image
                        Image.fromarray(img).save(output_path)
                        processed_files.append(output_path)
                        print(f"Processed: {input_path} -> {output_path}")
                    except Exception as e:
                        print(f"Error processing {input_path}: {str(e)}")
        
        return processed_files
    
    def augment_audio(self, audio, augmentation_type='noise'):
        """
        Apply data augmentation to audio
        
        Args:
            audio (np.array): Audio signal
            augmentation_type (str): Type of augmentation
            
        Returns:
            np.array: Augmented audio
        """
        if augmentation_type == 'noise':
            noise = np.random.randn(len(audio)) * 0.005
            return audio + noise
        elif augmentation_type == 'time_stretch':
            return librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
        elif augmentation_type == 'pitch_shift':
            return librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=np.random.randint(-2, 2)
            )
        else:
            return audio
