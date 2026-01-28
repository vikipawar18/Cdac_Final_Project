import tensorflow as tf
import os
import json

print(f"TensorFlow Version: {tf.__version__}")

model_path = 'mask_detector_model.h5'
if not os.path.exists(model_path):
    print(f"Model file {model_path} not found!")
else:
    print(f"\nAttempting to load {model_path}...")
    
    # Try 1: Standard load with compile=False
    try:
        print("\n[Try 1] Loading with compile=False...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✓ Model loaded successfully!")
        print(f"  Input shape: {model.input_shape}")
        print(f"  Output shape: {model.output_shape}")
    except Exception as e:
        print(f"✗ Failed: {str(e)[:100]}")
        
        # Try 2: Load with safe_mode=False
        try:
            print("\n[Try 2] Loading with compile=False and safe_mode=False...")
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            print("✓ Model loaded successfully!")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
        except Exception as e2:
            print(f"✗ Failed: {str(e2)[:100]}")
            
            # Try 3: Load weights only approach
            try:
                print("\n[Try 3] Attempting to extract model architecture...")
                import h5py
                with h5py.File(model_path, 'r') as f:
                    if 'model_config' in f.attrs:
                        model_config = f.attrs['model_config']
                        if isinstance(model_config, bytes):
                            model_config = model_config.decode('utf-8')
                        config_dict = json.loads(model_config)
                        print(f"  Model class: {config_dict.get('class_name', 'Unknown')}")
                        print(f"  Backend: {config_dict.get('backend', 'Unknown')}")
                    print("\n  Available keys in HDF5:", list(f.keys()))
            except Exception as e3:
                print(f"✗ Failed to inspect: {str(e3)[:100]}")
