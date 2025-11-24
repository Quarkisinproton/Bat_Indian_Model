"""
Models module for bat species classification
"""

from .bat_classifier import BatSpeciesClassifier

# Also import PyTorch models if they exist at parent level
# (workaround for import conflict between models.py and models/ folder)
try:
    # This won't work as intended since we're in the models package
    # Instead, we need to handle this at the package level
    pass
except ImportError:
    pass

__all__ = ['BatSpeciesClassifier']
