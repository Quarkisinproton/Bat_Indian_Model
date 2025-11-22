import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import BatDataset
from models import BatCNN, BatTransformer
import numpy as np

def train(args):
    # Device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset
    full_dataset = BatDataset(
        data_dir=args.data_dir,
        annotation_file=args.annotation_file,
        target_sample_rate=22050,
        duration=5.0
    )
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_classes = len(full_dataset.species_to_idx)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {full_dataset.species_to_idx}")

    # Model
    if args.model == 'cnn':
        model = BatCNN(num_classes=num_classes).to(device)
    elif args.model == 'transformer':
        # Input size depends on spectrogram settings. 
        # 5s * 22050Hz = 110250 samples. 
        # MelSpec (n_fft=1024, hop=512) -> Time steps approx 110250/512 = 215. 
        # n_mels=128.
        model = BatTransformer(num_classes=num_classes, input_height=128, input_width=216).to(device)
    
    # Loss and Optimizer
    criterion_cls = nn.CrossEntropyLoss()
    criterion_count = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, labels, counts) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            counts = counts.to(device).unsqueeze(1) # (B, 1)
            
            # Forward
            outputs_cls, outputs_count = model(images)
            
            loss_cls = criterion_cls(outputs_cls, labels)
            loss_count = criterion_count(outputs_count, counts)
            
            loss = loss_cls + 0.1 * loss_count # Weight count loss less? Or equal?
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation
        if (epoch + 1) % 5 == 0:
            evaluate(model, val_loader, device)

    print("Training finished.")
    torch.save(model.state_dict(), f"bat_model_{args.model}.pth")

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    count_error = 0
    
    with torch.no_grad():
        for images, labels, counts in loader:
            images = images.to(device)
            labels = labels.to(device)
            counts = counts.to(device).unsqueeze(1)
            
            outputs_cls, outputs_count = model(images)
            
            _, predicted = torch.max(outputs_cls.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            count_error += nn.functional.l1_loss(outputs_count, counts, reduction='sum').item()
            
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    print(f"Mean Count Error: {count_error / total:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gb/.gemini/antigravity/scratch/bat_project/data')
    parser.add_argument('--annotation_file', type=str, default='/home/gb/.gemini/antigravity/scratch/bat_project/data/annotations.json')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'transformer'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    train(args)
