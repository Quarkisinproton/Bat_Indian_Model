import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import BatDataset
from models import BatCNN, BatTransformer
from pathlib import Path
import numpy as np

def train(args):
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
    
    # Split dataset
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    if args.model == 'cnn':
        model = BatCNN(n_mels=args.n_mels, n_classes=2).to(device)
    elif args.model == 'transformer':
        model = BatTransformer(n_mels=args.n_mels, n_classes=2).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # Loss and Optimizer
    criterion_species = nn.CrossEntropyLoss()
    criterion_count = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_species_acc = 0.0
        train_count_mae = 0.0
        
        for batch in train_loader:
            spectrogram = batch['spectrogram'].to(device)
            species_label = batch['species_label'].to(device)
            call_count = batch['call_count'].to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            
            species_logits, count_pred = model(spectrogram)
            
            loss_s = criterion_species(species_logits, species_label)
            loss_c = criterion_count(count_pred, call_count)
            
            loss = loss_s + args.count_weight * loss_c
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Metrics
            preds = torch.argmax(species_logits, dim=1)
            train_species_acc += (preds == species_label).sum().item()
            train_count_mae += torch.abs(count_pred - call_count).sum().item()
            
        train_loss /= len(train_loader)
        train_species_acc /= len(train_dataset)
        train_count_mae /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_species_acc = 0.0
        val_count_mae = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                spectrogram = batch['spectrogram'].to(device)
                species_label = batch['species_label'].to(device)
                call_count = batch['call_count'].to(device).float().unsqueeze(1)
                
                species_logits, count_pred = model(spectrogram)
                
                loss_s = criterion_species(species_logits, species_label)
                loss_c = criterion_count(count_pred, call_count)
                
                loss = loss_s + args.count_weight * loss_c
                val_loss += loss.item()
                
                preds = torch.argmax(species_logits, dim=1)
                val_species_acc += (preds == species_label).sum().item()
                val_count_mae += torch.abs(count_pred - call_count).sum().item()
                
        val_loss /= len(val_loader)
        val_species_acc /= len(val_dataset)
        val_count_mae /= len(val_dataset)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_species_acc:.2f} MAE: {train_count_mae:.2f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_species_acc:.2f} MAE: {val_count_mae:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{args.model}.pth")
            print("Saved best model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bat Species Model")
    parser.add_argument("--annotations", type=str, default="../data/annotations.json", help="Path to annotations JSON")
    parser.add_argument("--data_dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "transformer"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sample_rate", type=int, default=384000, help="Target sample rate")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of mel bins")
    parser.add_argument("--duration", type=float, default=5.0, help="Fixed duration in seconds")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--count_weight", type=float, default=1.0, help="Weight for count loss")
    
    args = parser.parse_args()
    train(args)
