"""
Training script for Bat Species Classification Model
"""

import argparse
import os
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.bat_classifier import BatSpeciesClassifier
from src.utils.data_loader import BatDataLoader, create_data_generators


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Bat Species Classification Model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/sample',
                       help='Directory containing processed spectrogram data (defaults to data/sample)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                       help='Image size (height width)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='cnn',
                       choices=['cnn', 'efficientnet', 'resnet'],
                       help='Type of model architecture')
    parser.add_argument('--use_pretrained', action='store_true',
                       help='Use pretrained weights for transfer learning')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer type')
    
    # Data split parameters
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Proportion of training data for validation')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    return parser.parse_args()


def setup_callbacks(checkpoint_dir, model_name):
    """
    Setup training callbacks
    
    Args:
        checkpoint_dir (str): Directory for checkpoints
        model_name (str): Name of the model
        
    Returns:
        list: List of callbacks
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    callbacks = []
    
    # Model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.h5')
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard
    log_dir = os.path.join('logs', model_name, datetime.now().strftime('%Y%m%d-%H%M%S'))
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard_callback)
    
    return callbacks


def train_model(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Set random seeds for reproducibility
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    # Configure TensorFlow to allow GPU memory growth when running on Colab
    # This helps avoid TF pre-allocating all GPU memory and reduces OOM issues
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except Exception as e:
        print(f"Could not set TensorFlow GPU memory growth: {e}")
    
    print("=" * 50)
    print("Bat Species Classification - Training")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    img_size_2d = tuple(args.img_size)  # 2D size for data loader
    img_size = img_size_2d + (3,)  # 3D size with channel dimension for model
    data_loader = BatDataLoader(args.data_dir, img_size=img_size_2d)
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_seed
    )
    
    num_classes = data_loader.get_num_classes()
    
    # Create data generators
    print("\nCreating data generators...")
    train_gen, val_gen = create_data_generators(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        augment=args.augment
    )
    
    # Build model
    print(f"\nBuilding {args.model_type} model...")
    classifier = BatSpeciesClassifier(
        num_classes=num_classes,
        input_shape=img_size,
        model_type=args.model_type,
        use_pretrained=args.use_pretrained
    )
    
    classifier.build_model()
    classifier.compile_model(
        learning_rate=args.learning_rate,
        optimizer=args.optimizer
    )
    
    print("\nModel Summary:")
    classifier.get_model_summary()
    
    # Setup callbacks
    model_name = f"bat_classifier_{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    callbacks = setup_callbacks(args.checkpoint_dir, model_name)
    
    # Train model
    print("\nStarting training...")
    history = classifier.model.fit(
        train_gen,
        epochs=args.epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = classifier.model.evaluate(X_test, y_test, verbose=0)
    
    # Extract results (first is loss, rest are metrics)
    test_loss = test_results[0]
    test_acc = test_results[1] if len(test_results) > 1 else 0.0
    test_precision = test_results[2] if len(test_results) > 2 else 0.0
    test_recall = test_results[3] if len(test_results) > 3 else 0.0
    test_auc = test_results[4] if len(test_results) > 4 else 0.0
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    if len(test_results) > 1:
        print(f"  Accuracy: {test_acc:.4f}")
    if len(test_results) > 2:
        print(f"  Precision: {test_precision:.4f}")
    if len(test_results) > 3:
        print(f"  Recall: {test_recall:.4f}")
    if len(test_results) > 4:
        print(f"  AUC: {test_auc:.4f}")
    
    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'{model_name}.h5')
    classifier.save_model(model_path)
    
    # Save label encoder
    encoder_path = os.path.join(args.output_dir, f'{model_name}_labels.json')
    data_loader.save_label_encoder(encoder_path)
    
    # Save training history
    history_path = os.path.join(args.output_dir, f'{model_name}_history.json')
    history_dict = {key: [float(val) for val in values] 
                   for key, values in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    
    # Save training configuration
    config_path = os.path.join(args.output_dir, f'{model_name}_config.json')
    config = vars(args)
    config['num_classes'] = num_classes
    config['class_names'] = data_loader.get_class_names()
    config['test_accuracy'] = float(test_acc)
    config['test_loss'] = float(test_loss)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Labels saved to: {encoder_path}")
    print(f"Configuration saved to: {config_path}")


if __name__ == '__main__':
    args = parse_args()
    train_model(args)
