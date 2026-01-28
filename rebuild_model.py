"""
Model Rebuilder Script
Rebuilds the model using the known architecture and loads weights
"""
import tensorflow as tf
from tensorflow.keras import layers, models
import os

print(f"TensorFlow Version: {tf.__version__}")

def build_model(seq_length=20):
    """Build the deepfake detection model architecture"""
    augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1)
    ])

    base_cnn = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_cnn.trainable = False 

    model = models.Sequential([
        layers.Input(shape=(seq_length, 128, 128, 3)),
        layers.TimeDistributed(augmentation),
        layers.TimeDistributed(base_cnn),
        layers.LSTM(128, dropout=0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    return model

old_model_path = 'mask_detector_model.h5'
new_model_path = 'deepfake_model_rebuilt.h5'

print(f"\nRebuilding model architecture...")
model = build_model()
print("✓ Model architecture created")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

if os.path.exists(old_model_path):
    try:
        print(f"\nAttempting to load weights from {old_model_path}...")
        model.load_weights(old_model_path)
        print("✓ Weights loaded successfully!")
        
        print(f"\nSaving rebuilt model to {new_model_path}...")
        model.save(new_model_path)
        print(f"✓ Model saved successfully!")
        
        # Verify the new model
        print("\nVerifying new model...")
        test_model = tf.keras.models.load_model(new_model_path, compile=False)
        print(f"✓ New model loads successfully!")
        print(f"  Input shape: {test_model.input_shape}")
        print(f"  Output shape: {test_model.output_shape}")
        
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS! Use '{new_model_path}' in your application.")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ Failed to load weights: {e}")
        print("\nSaving model with random weights for testing...")
        model.save(new_model_path)
        print(f"✓ Model saved to {new_model_path} (with random weights)")
        print("\nNote: This model has random weights and won't make accurate predictions.")
        print("You'll need to retrain or provide the correct weights file.")
else:
    print(f"\n✗ Model file {old_model_path} not found!")
    print("\nSaving model with random weights...")
    model.save(new_model_path)
    print(f"✓ Model saved to {new_model_path} (with random weights)")
