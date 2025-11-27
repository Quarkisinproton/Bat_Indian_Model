import torch
import torch.nn as nn
import math

class BatCNN(nn.Module):
    def __init__(self, n_mels=128, n_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.flatten = nn.Flatten()
        self.fc_shared = nn.Linear(64 * 4 * 4, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Single head: only species classification (drop count prediction)
        self.species_head = nn.Linear(128, n_classes)

    def forward(self, x):
        # x: (Batch, 1, features, time)
        # features = mel_spec (128) + mfcc (40) + spectral_centroid (1) + bandwidth (1) + zcr (1) = 171
        x = self.features(x)
        x = self.flatten(x)
        x = self.relu(self.fc_shared(x))
        x = self.dropout(x)
        
        species_logits = self.species_head(x)
        
        return species_logits

class BatTransformer(nn.Module):
    def __init__(self, n_mels=128, max_len=1000, n_classes=2, d_model=256, nhead=4, num_layers=4, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding
        # Input: (Batch, 1, n_mels, time)
        # We assume n_mels is divisible by patch_size[0] and time by patch_size[1]
        # For simplicity, let's treat n_mels as the feature dimension and time as sequence
        # But standard ViT patches 2D.
        # Let's use a Conv2d to create patches
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_shared = nn.Linear(d_model, 512)
        self.relu = nn.ReLU()
        
        # Heads
        self.species_head = nn.Linear(512, n_classes)
        self.count_head = nn.Linear(512, 1)

    def forward(self, x):
        # x: (Batch, 1, n_mels, time)
        # Patchify
        x = self.patch_embed(x) # (Batch, d_model, H', W')
        x = x.flatten(2).transpose(1, 2) # (Batch, Seq, d_model)
        
        B, S, D = x.shape
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add Positional Embedding (truncate or interpolate if needed, here we assume max_len is enough)
        if x.shape[1] > self.pos_embed.shape[1]:
             x = x[:, :self.pos_embed.shape[1], :]
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        x = self.transformer(x)
        
        # Use CLS token for prediction
        cls_out = x[:, 0, :]
        
        feat = self.relu(self.fc_shared(cls_out))
        
        species_logits = self.species_head(feat)
        count_pred = self.count_head(feat)
        
        return species_logits, count_pred
