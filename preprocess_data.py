"""
Script to preprocess audio data for training
Converts audio files to spectrograms
"""

import argparse
import os
from src.preprocessing.audio_processor import AudioProcessor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Preprocess audio data for bat species classification')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing audio files organized by species')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed spectrograms')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                       help='Output image size (height width)')
    parser.add_argument('--sample_rate', type=int, default=44100,
                       help='Audio sample rate')
    parser.add_argument('--n_mels', type=int, default=128,
                       help='Number of mel bands')
    parser.add_argument('--duration', type=float, default=5.0,
                       help='Duration of audio segments in seconds')
    
    return parser.parse_args()


def main():
    """Main preprocessing function"""
    args = parse_args()
    
    print("=" * 50)
    print("Bat Species Audio Preprocessing")
    print("=" * 50)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Initialize audio processor
    print("\nInitializing audio processor...")
    processor = AudioProcessor(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        duration=args.duration
    )
    
    # Process all audio files
    print(f"\nProcessing audio files from: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output image size: {args.img_size}")
    
    processed_files = processor.batch_process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        output_size=tuple(args.img_size)
    )
    
    print("\n" + "=" * 50)
    print(f"Preprocessing complete!")
    print(f"Total files processed: {len(processed_files)}")
    print("=" * 50)


if __name__ == '__main__':
    main()
