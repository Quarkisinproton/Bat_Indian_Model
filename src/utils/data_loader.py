"""
Data loading and management utilities
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from PIL import Image
import json


class BatDataLoader:
    """
    Load and manage bat species dataset
    """
    
    def __init__(self, data_dir, img_size=(128, 128)):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Directory containing organized data
            img_size (tuple): Image size for loading
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.label_encoder = LabelEncoder()
        self.class_names = []
        
    def load_data_from_directory(self, directory):
        """
        Load images and labels from directory structure
        Expected structure: directory/class_name/image_files
        
        Args:
            directory (str): Root directory containing class folders
            
        Returns:
            tuple: (images, labels)
        """
        images = []
        labels = []
        
        if not os.path.exists(directory):
            raise ValueError(f"Directory not found: {directory}")
        
        # Get class names from subdirectories
        class_names = sorted([d for d in os.listdir(directory) 
                            if os.path.isdir(os.path.join(directory, d))])
        
        if not class_names:
            raise ValueError(f"No class directories found in {directory}")
        
        self.class_names = class_names
        
        for class_name in class_names:
            class_dir = os.path.join(directory, class_name)
            
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, filename)
                    
                    try:
                        # Load and preprocess image
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                        
                        images.append(img_array)
                        labels.append(class_name)
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels)
    
    def prepare_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """
        Load and split data into train, validation, and test sets
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            random_state (int): Random seed
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Load all data
        X, y = self.load_data_from_directory(self.data_dir)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split into train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=random_state, stratify=y_encoded
        )
        
        # Split train into train and validation
        val_ratio = val_size / (1 - test_size)
        y_temp_labels = np.argmax(y_temp, axis=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp_labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Classes: {self.class_names}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_names(self):
        """
        Get list of class names
        
        Returns:
            list: Class names
        """
        return self.class_names
    
    def get_num_classes(self):
        """
        Get number of classes
        
        Returns:
            int: Number of classes
        """
        return len(self.class_names)
    
    def decode_predictions(self, predictions, top_k=3):
        """
        Decode model predictions to class names
        
        Args:
            predictions (np.array): Model predictions
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of (class_name, probability) tuples
        """
        results = []
        
        for pred in predictions:
            top_indices = np.argsort(pred)[-top_k:][::-1]
            top_probs = pred[top_indices]
            top_classes = [self.class_names[i] for i in top_indices]
            
            results.append(list(zip(top_classes, top_probs)))
        
        return results
    
    def save_label_encoder(self, filepath):
        """
        Save label encoder and class names
        
        Args:
            filepath (str): Path to save the encoder
        """
        data = {
            'classes': self.label_encoder.classes_.tolist(),
            'class_names': self.class_names
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Label encoder saved to {filepath}")
    
    def load_label_encoder(self, filepath):
        """
        Load label encoder and class names
        
        Args:
            filepath (str): Path to load the encoder
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.label_encoder.classes_ = np.array(data['classes'])
        self.class_names = data['class_names']
        
        print(f"Label encoder loaded from {filepath}")


def create_data_generators(X_train, y_train, X_val, y_val, batch_size=32, augment=True):
    """
    Create data generators with optional augmentation
    
    Args:
        X_train (np.array): Training images
        y_train (np.array): Training labels
        X_val (np.array): Validation images
        y_val (np.array): Validation labels
        batch_size (int): Batch size
        augment (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (train_generator, val_generator)
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    if augment:
        train_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator()
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train, y_train,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_generator = val_datagen.flow(
        X_val, y_val,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_generator, val_generator
