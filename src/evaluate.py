import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import BatDataset
from models import BatCNN, BatTransformer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset
    dataset = BatDataset(
        annotations_file=args.annotations,
        data_dir=args.data_dir,
        target_sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        fixed_duration=args.duration
    )
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    if args.model == 'cnn':
        model = BatCNN(n_mels=args.n_mels, n_classes=2).to(device)
    elif args.model == 'transformer':
        model = BatTransformer(n_mels=args.n_mels, n_classes=2).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_count_preds = []
    all_count_targets = []
    
    with torch.no_grad():
        for batch in loader:
            spectrogram = batch['spectrogram'].to(device)
            species_label = batch['species_label'].to(device)
            call_count = batch['call_count'].to(device).float().unsqueeze(1)
            
            species_logits, count_pred = model(spectrogram)
            
            preds = torch.argmax(species_logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(species_label.cpu().numpy())
            all_count_preds.extend(count_pred.cpu().numpy())
            all_count_targets.extend(call_count.cpu().numpy())
            
    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Species 0", "Species 1"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    mae = np.mean(np.abs(np.array(all_count_preds) - np.array(all_count_targets)))
    print(f"\nCount MAE: {mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bat Species Model")
    parser.add_argument("--annotations", type=str, default="../data/annotations.json", help="Path to annotations JSON")
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "transformer"], help="Model architecture")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--sample_rate", type=int, default=384000, help="Target sample rate")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--duration", type=float, default=5.0, help="Fixed duration in seconds")
    
    args = parser.parse_args()
    evaluate(args)
