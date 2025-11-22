"""
Evaluation script for Bat Species Classification Model
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from src.utils.data_loader import BatDataLoader


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate Bat Species Classification Model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.h5)')
    parser.add_argument('--labels', type=str, required=True,
                       help='Path to labels file (.json)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing test data')
    parser.add_argument('--img_size', type=int, nargs=2, default=[128, 128],
                       help='Image size (height width)')
    parser.add_argument('--output_dir', type=str, default='evaluation',
                       help='Directory to save evaluation results')
    
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix
    
    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
        output_path (str): Path to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_per_class_metrics(metrics_dict, output_path):
    """
    Plot per-class precision, recall, and F1-score
    
    Args:
        metrics_dict (dict): Dictionary with metrics per class
        output_path (str): Path to save plot
    """
    classes = list(metrics_dict.keys())
    precision = [metrics_dict[c]['precision'] for c in classes]
    recall = [metrics_dict[c]['recall'] for c in classes]
    f1 = [metrics_dict[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Species')
    plt.ylabel('Score')
    plt.title('Per-Class Performance Metrics')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.ylim([0, 1.05])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to {output_path}")


def evaluate(args):
    """
    Main evaluation function
    
    Args:
        args: Command line arguments
    """
    print("=" * 50)
    print("Bat Species Classification - Evaluation")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = keras.models.load_model(args.model)
    
    # Load labels
    print(f"Loading labels from {args.labels}...")
    with open(args.labels, 'r') as f:
        label_data = json.load(f)
    class_names = label_data['class_names']
    
    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    img_size = tuple(args.img_size)
    data_loader = BatDataLoader(args.data_dir, img_size=img_size)
    
    X, y = data_loader.load_data_from_directory(args.data_dir)
    print(f"Loaded {len(X)} test samples")
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.utils import to_categorical
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(label_data['classes'])
    y_encoded = label_encoder.transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_categorical, axis=1)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # Overall metrics
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )
    
    # Weighted averages
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision_avg:.4f}")
    print(f"Weighted Recall: {recall_avg:.4f}")
    print(f"Weighted F1-Score: {f1_avg:.4f}")
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    metrics_dict = {}
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<30} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")
        metrics_dict[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1-score': float(f1[i]),
            'support': int(support[i])
        }
    
    # Classification report
    print("\n" + "=" * 50)
    print("Detailed Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Save results
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'precision': float(precision_avg),
            'recall': float(recall_avg),
            'f1_score': float(f1_avg)
        },
        'per_class': metrics_dict,
        'confusion_matrix': cm.tolist()
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Plot per-class metrics
    metrics_path = os.path.join(args.output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(metrics_dict, metrics_path)
    
    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
