"""
Example usage of the Bat Species Classification system
"""

import numpy as np
from src.models.bat_classifier import BatSpeciesClassifier
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.data_loader import BatDataLoader, create_data_generators


def example_audio_processing():
    """Example: Process audio file to spectrogram"""
    print("=" * 50)
    print("Example 1: Audio Processing")
    print("=" * 50)
    
    # Initialize audio processor
    processor = AudioProcessor(
        sample_rate=44100,
        n_mels=128,
        duration=5.0
    )
    
    print("\nAudio Processor initialized with:")
    print(f"  Sample Rate: {processor.sample_rate} Hz")
    print(f"  Mel Bands: {processor.n_mels}")
    print(f"  Duration: {processor.duration} seconds")
    
    # Example: Process a single audio file (if it exists)
    # audio_path = 'data/sample/bat_call.wav'
    # if os.path.exists(audio_path):
    #     img = processor.process_audio_file(audio_path, output_size=(128, 128))
    #     print(f"\nProcessed audio to spectrogram: {img.shape}")
    
    print("\nTo process audio files:")
    print("  processor.process_audio_file('audio.wav', output_size=(128, 128))")


def example_model_building():
    """Example: Build and compile model"""
    print("\n" + "=" * 50)
    print("Example 2: Model Building")
    print("=" * 50)
    
    # Create classifier
    classifier = BatSpeciesClassifier(
        num_classes=10,
        input_shape=(128, 128, 3),
        model_type='cnn'
    )
    
    # Build model
    classifier.build_model()
    
    # Compile model
    classifier.compile_model(learning_rate=0.001, optimizer='adam')
    
    print("\nModel built and compiled successfully!")
    print(f"  Model Type: {classifier.model_type}")
    print(f"  Number of Classes: {classifier.num_classes}")
    print(f"  Input Shape: {classifier.input_shape}")
    
    # Get model summary
    print("\nModel Architecture:")
    classifier.get_model_summary()


def example_data_loading():
    """Example: Load and prepare data"""
    print("\n" + "=" * 50)
    print("Example 3: Data Loading")
    print("=" * 50)
    
    print("\nTo load data, organize it as:")
    print("  data/processed/")
    print("    ├── Species_1/")
    print("    │   ├── image1.png")
    print("    │   └── image2.png")
    print("    └── Species_2/")
    print("        └── image1.png")
    
    print("\nThen use:")
    print("  data_loader = BatDataLoader('data/processed', img_size=(128, 128))")
    print("  X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()")


def example_training():
    """Example: Training workflow"""
    print("\n" + "=" * 50)
    print("Example 4: Training Workflow")
    print("=" * 50)
    
    print("\nComplete training workflow:")
    print("""
# 1. Initialize components
classifier = BatSpeciesClassifier(num_classes=10, model_type='cnn')
data_loader = BatDataLoader('data/processed')

# 2. Load data
X_train, X_val, X_test, y_train, y_val, y_test = data_loader.prepare_data()

# 3. Build and compile model
classifier.build_model()
classifier.compile_model(learning_rate=0.001)

# 4. Create data generators with augmentation
train_gen, val_gen = create_data_generators(
    X_train, y_train, X_val, y_val,
    batch_size=32,
    augment=True
)

# 5. Train model
history = classifier.model.fit(
    train_gen,
    epochs=50,
    validation_data=val_gen
)

# 6. Evaluate
test_loss, test_acc = classifier.model.evaluate(X_test, y_test)

# 7. Save model
classifier.save_model('models/my_bat_classifier.h5')
    """)


def example_prediction():
    """Example: Making predictions"""
    print("\n" + "=" * 50)
    print("Example 5: Making Predictions")
    print("=" * 50)
    
    print("\nTo make predictions on new data:")
    print("""
# 1. Load trained model
from tensorflow import keras
model = keras.models.load_model('models/my_bat_classifier.h5')

# 2. Process input (audio or image)
processor = AudioProcessor()
img = processor.process_audio_file('new_audio.wav', output_size=(128, 128))
img = img / 255.0  # Normalize
img = np.expand_dims(img, axis=0)  # Add batch dimension

# 3. Predict
predictions = model.predict(img)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

print(f"Predicted class: {predicted_class} with confidence: {confidence:.2%}")
    """)


def example_transfer_learning():
    """Example: Transfer learning with pretrained models"""
    print("\n" + "=" * 50)
    print("Example 6: Transfer Learning")
    print("=" * 50)
    
    print("\nUsing pretrained models (EfficientNet or ResNet):")
    print("""
# 1. Create classifier with pretrained weights
classifier = BatSpeciesClassifier(
    num_classes=10,
    model_type='efficientnet',
    use_pretrained=True
)

# 2. Build and compile
classifier.build_model()
classifier.compile_model(learning_rate=0.0001)  # Lower learning rate

# 3. Train with frozen base layers
history = classifier.model.fit(X_train, y_train, epochs=20)

# 4. Unfreeze and fine-tune
classifier.unfreeze_base_model(num_layers=50)
classifier.compile_model(learning_rate=0.00001)  # Even lower learning rate

# 5. Continue training
history = classifier.model.fit(X_train, y_train, epochs=30)
    """)


def main():
    """Run all examples"""
    print("╔" + "=" * 58 + "╗")
    print("║  Bat Species Classification - Usage Examples           ║")
    print("╚" + "=" * 58 + "╝")
    
    example_audio_processing()
    example_model_building()
    example_data_loading()
    example_training()
    example_prediction()
    example_transfer_learning()
    
    print("\n" + "=" * 60)
    print("For more detailed examples, see:")
    print("  - README.md: Comprehensive documentation")
    print("  - USAGE.md: Detailed usage guide")
    print("  - Command line scripts: train.py, predict.py, etc.")
    print("=" * 60)


if __name__ == '__main__':
    main()
