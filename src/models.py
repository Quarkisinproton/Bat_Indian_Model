import torch
import torch.nn as nn
import torch.nn.functional as F

class BatCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(BatCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Heads
        self.fc_species = nn.Linear(128, num_classes)
        self.fc_count = nn.Linear(128, 1) # Regression for count

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        species_logits = self.fc_species(x)
        count_pred = self.fc_count(x)
        
        return species_logits, count_pred

class BatTransformer(nn.Module):
    def __init__(self, num_classes, input_height=128, input_width=215, patch_size=16, embed_dim=128, num_heads=4, num_layers=4):
        super(BatTransformer, self).__init__()
        # Simplified ViT-like architecture
        self.patch_size = patch_size
        self.num_patches = (input_height // patch_size) * (input_width // patch_size)
        self.embed_dim = embed_dim
        
        self.projection = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Heads
        self.fc_species = nn.Linear(embed_dim, num_classes)
        self.fc_count = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (B, 1, H, W)
        B = x.shape[0]
        
        # Patch embedding
        x = self.projection(x) # (B, E, H', W')
        x = x.flatten(2).transpose(1, 2) # (B, N, E)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        # Note: This simple implementation assumes fixed input size. 
        # For variable size, interpolation of pos_embed is needed.
        if x.shape[1] == self.pos_embed.shape[1]:
             x = x + self.pos_embed
        else:
             # Handle size mismatch if necessary (e.g. crop or interpolate)
             # For now, we assume input size matches or we slice pos_embed (risky)
             x = x + self.pos_embed[:, :x.shape[1], :]

        x = self.transformer(x)
        
        # Use CLS token for prediction
        cls_out = x[:, 0]
        
        species_logits = self.fc_species(cls_out)
        count_pred = self.fc_count(cls_out)
        
        return species_logits, count_pred
