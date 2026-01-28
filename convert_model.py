"""
Model Converter Script
Converts the old TensorFlow model to a compatible format for TensorFlow 2.20.0
"""
import tensorflow as tf
import h5py
import json
import numpy as np

print(f"TensorFlow Version: {tf.__version__}")

old_model_path = 'mask_detector_model.h5'
new_model_path = 'mask_detector_model_v2.h5'

print(f"\nConverting {old_model_path} to {new_model_path}...")

try:
    # Try to load with custom handling
    print("\nAttempting to load model architecture...")
    
    with h5py.File(old_model_path, 'r') as f:
        # Get model config
        if 'model_config' in f.attrs:
            model_config = f.attrs['model_config']
            if isinstance(model_config, bytes):
                model_config = model_config.decode('utf-8')
            
            config_dict = json.loads(model_config)
            print(f"Model class: {config_dict.get('class_name')}")
            
            # Try to reconstruct from config with modifications
            # Remove problematic 'groups' parameter from DepthwiseConv2D layers
            def fix_config(config):
                if isinstance(config, dict):
                    if config.get('class_name') == 'DepthwiseConv2D':
                        if 'config' in config and 'groups' in config['config']:
                            del config['config']['groups']
                            print("  Fixed DepthwiseConv2D layer")
                    
                    for key, value in config.items():
                        config[key] = fix_config(value)
                
                elif isinstance(config, list):
                    return [fix_config(item) for item in config]
                
                return config
            
            fixed_config = fix_config(config_dict)
            
            # Reconstruct model from fixed config
            print("\nReconstructing model from fixed configuration...")
            model = tf.keras.Model.from_config(fixed_config['config'])
            
            # Load weights
            print("Loading weights...")
            with h5py.File(old_model_path, 'r') as f_weights:
                if 'model_weights' in f_weights:
                    # Use load_weights_from_hdf5_group
                    from tensorflow.python.keras.saving import hdf5_format
                    hdf5_format.load_weights_from_hdf5_group(f_weights['model_weights'], model.layers)
                    print("✓ Weights loaded successfully")
            
            # Save in new format
            print(f"\nSaving converted model to {new_model_path}...")
            model.save(new_model_path)
            print(f"✓ Model saved successfully!")
            
            # Verify the new model
            print("\nVerifying new model...")
            test_model = tf.keras.models.load_model(new_model_path, compile=False)
            print(f"✓ New model loads successfully!")
            print(f"  Input shape: {test_model.input_shape}")
            print(f"  Output shape: {test_model.output_shape}")
            
            print(f"\n✓ Conversion complete! Use '{new_model_path}' in your application.")
            
except Exception as e:
    print(f"\n✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALTERNATIVE SOLUTION:")
    print("="*60)
    print("The model file appears to be incompatible with TensorFlow 2.20.0.")
    print("\nOptions:")
    print("1. Retrain the model with the current TensorFlow version")
    print("2. Downgrade TensorFlow to match the version used to create the model")
    print("3. Use TensorFlow Lite converter to create a .tflite model")
    print("4. Export and reimport using SavedModel format instead of HDF5")
