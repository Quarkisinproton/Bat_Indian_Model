"""
Bat Species Classifier Model
A deep learning model for classifying Indian bat species from audio spectrograms
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50
import numpy as np


class BatSpeciesClassifier:
    """
    Deep Learning model for bat species classification
    """
    
    def __init__(self, num_classes=10, input_shape=(128, 128, 3), 
                 model_type='cnn', use_pretrained=False):
        """
        Initialize the bat species classifier
        
        Args:
            num_classes (int): Number of bat species to classify
            input_shape (tuple): Input shape for spectrograms (height, width, channels)
            model_type (str): Type of model ('cnn', 'efficientnet', 'resnet')
            use_pretrained (bool): Whether to use pretrained weights
        """
        """PyTorch implementation of the Bat Species Classifier.

        This module provides a light wrapper `BatSpeciesClassifier` that builds a
        PyTorch model (either a small custom CNN, ResNet50 or EfficientNet-B0 backbone)
        and convenient helpers for saving/loading and running inference. The API is
        kept intentionally simple so it integrates into the project's existing
        training/evaluation scripts. The implementation is compatible with Google
        Colab (install PyTorch via the repository `requirements.txt`).
        """

        from typing import Optional, Tuple
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import torchvision.models as tvmodels
        import numpy as np


        def _make_simple_cnn(num_classes: int, in_channels: int = 3) -> nn.Module:
            """Small CNN suitable for spectrogram inputs.

            Args:
                num_classes: number of output classes
                in_channels: input channels (usually 1 or 3)
            """
            return nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(256, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
            )


        class BatSpeciesClassifier:
            """Wrapper for PyTorch models for bat species classification.

            Usage:
                cls = BatSpeciesClassifier(num_classes=10, input_shape=(128,128,3))
                model = cls.build_model('cnn')
                model.to(device)

            The class provides `save_model`, `load_model` and `predict` helpers to make
            it easy to integrate with existing scripts and to run inference on Colab.
            """

            def __init__(
                self,
                num_classes: int = 10,
                input_shape: Tuple[int, int, int] = (128, 128, 3),
                model_type: str = 'cnn',
                use_pretrained: bool = False,
                device: Optional[torch.device] = None,
            ):
                self.num_classes = num_classes
                self.input_shape = input_shape
                self.model_type = model_type
                self.use_pretrained = use_pretrained
                self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                self.model: Optional[nn.Module] = None

            def build_model(self) -> nn.Module:
                """Build and return the requested model.

                model_type: 'cnn' | 'resnet' | 'efficientnet'
                """
                c = self.input_shape[2] if len(self.input_shape) == 3 else 1

                if self.model_type == 'cnn':
                    self.model = _make_simple_cnn(self.num_classes, in_channels=c)

                elif self.model_type == 'resnet':
                    # use torchvision resnet50
                    backbone = tvmodels.resnet50(pretrained=self.use_pretrained)
                    # replace first conv if channel mismatch
                    if c != 3:
                        backbone.conv1 = nn.Conv2d(c, 64, kernel_size=7, stride=2, padding=3, bias=False)
                    # remove fc and avgpool, use adaptive pooling
                    backbone.fc = nn.Linear(backbone.fc.in_features, self.num_classes)
                    self.model = backbone

                elif self.model_type == 'efficientnet':
                    # torchvision's efficientnet_b0
                    try:
                        backbone = tvmodels.efficientnet_b0(pretrained=self.use_pretrained)
                    except Exception:
                        # older torchvision may have different API; try hub fallback
                        backbone = tvmodels.efficientnet_b0(pretrained=self.use_pretrained)
                    if c != 3:
                        # adjust first conv
                        backbone.features[0][0] = nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1, bias=False)
                    # replace classifier
                    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, self.num_classes)
                    self.model = backbone

                else:
                    raise ValueError(f"Unknown model_type: {self.model_type}")

                return self.model

            def to(self, device: Optional[torch.device] = None) -> nn.Module:
                """Move model to device and return it."""
                if self.model is None:
                    self.build_model()
                if device is None:
                    device = self.device
                self.model.to(device)
                self.device = device
                return self.model

            def save_model(self, filepath: str) -> None:
                """Save model state_dict to a file."""
                if self.model is None:
                    raise RuntimeError("No model to save")
                torch.save({'model_state': self.model.state_dict(), 'model_type': self.model_type}, filepath)

            def load_model(self, filepath: str, map_location: Optional[torch.device] = None) -> nn.Module:
                """Load model state_dict from a file and return the model on device."""
                ckpt = torch.load(filepath, map_location=map_location or self.device)
                model_type = ckpt.get('model_type', self.model_type)
                self.model_type = model_type
                self.build_model()
                self.model.load_state_dict(ckpt['model_state'])
                self.to(map_location or self.device)
                return self.model

            def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
                """Run inference on numpy array inputs and return probabilities.

                Accepts X shaped (N, H, W, C) or (N, C, H, W). Returns numpy array (N, num_classes).
                """
                if self.model is None:
                    raise RuntimeError("Model not built or loaded")

                self.model.eval()
                # convert to tensor
                if isinstance(X, np.ndarray):
                    x = torch.from_numpy(X)
                else:
                    x = X
                # ensure float and channel-first
                if x.ndim == 4 and x.shape[-1] == self.input_shape[2]:
                    x = x.permute(0, 3, 1, 2)
                x = x.float()

                preds = []
                with torch.no_grad():
                    for i in range(0, x.shape[0], batch_size):
                        xb = x[i:i+batch_size].to(self.device)
                        out = self.model(xb)
                        probs = F.softmax(out, dim=1).cpu().numpy()
                        preds.append(probs)

                return np.vstack(preds)

            def get_num_parameters(self) -> int:
                if self.model is None:
                    self.build_model()
                return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
