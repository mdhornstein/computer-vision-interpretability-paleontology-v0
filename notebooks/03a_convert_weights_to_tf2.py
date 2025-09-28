#!/usr/bin/env python3
"""
Simple TensorFlow 1.x to SavedModel Converter
Converts TF1 checkpoint to modern SavedModel format with validation
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add parent directory to path to find local tensorflow modules
# Script is in /workspace/notebooks/, tensorflow/ is in /workspace/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now import from local structure
import tensorflow.contrib.slim as slim
from nets import inception_resnet_v2

# Ensure TF1 behavior
tf.compat.v1.disable_v2_behavior()

# Enable resource variables to avoid float32_ref warnings in TF2
# Necessary if you want to further train the model 
tf.compat.v1.enable_resource_variables()

# --- Configuration ---
CHECKPOINT_DIR = "/workspace/models/fid_classification/tf1/" 
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "model.ckpt-60")
EXPORT_DIR = "/workspace/models/fid_classification/tf1_from_community_allow_train/1" 

# Model parameters (must match original training)
N_CLASSES = 50
HEIGHT = 299
WIDTH = 299

def check_checkpoint_exists(checkpoint_path):
    """Verify checkpoint files exist"""
    required_files = [
        checkpoint_path + ".meta",
        checkpoint_path + ".index", 
        checkpoint_path + ".data-00000-of-00001"
    ]
    
    print("Checking checkpoint files:")
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print("  {} {}".format(status, file_path))
        if not exists:
            return False
    return True

def test_model_inference(sess, input_tensor, predictions):
    """Test model with dummy input to verify it works"""
    print("\n--- Testing Model ---")
    try:
        # Create random test input
        dummy_input = np.random.rand(1, HEIGHT, WIDTH, 3).astype(np.float32)
        
        # Run inference
        pred_output = sess.run(predictions, {input_tensor: dummy_input})
        
        print("✅ Model test successful!")
        print("   Input shape: {}".format(dummy_input.shape))
        print("   Output shape: {}".format(pred_output.shape))
        print("   Prediction sum: {:.6f} (should be ~1.0)".format(np.sum(pred_output[0])))
        print("   Top class: {} (confidence: {:.4f})".format(np.argmax(pred_output[0]), np.max(pred_output[0])))
        
        return True
    except Exception as e:
        print("❌ Model test failed: {}".format(e))
        return False

def main():
    print("TensorFlow 1.x to SavedModel Converter")
    print("TensorFlow version: {}".format(tf.__version__))
    print("=" * 50)

    # Check checkpoint exists
    if not check_checkpoint_exists(CHECKPOINT_FILE):
        print("❌ Checkpoint files missing. Please check the path.")
        return False

    # Reset graph and create session
    # tf.reset_default_graph() -- check this 
    tf.compat.v1.reset_default_graph()
    
    with tf.Session() as sess:
        print("\n--- Building Model ---")
        
        # Create input placeholder
        input_tensor = tf.placeholder(tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='input')

        # Build Inception ResNet V2 architecture  
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
            logits, endpoints = inception_resnet_v2.inception_resnet_v2(
                input_tensor,
                num_classes=N_CLASSES,
                is_training=False  # Important: set to False for inference
            )
        
        # Add softmax predictions
        predictions = tf.nn.softmax(logits, name='predictions')
        
        print("✅ Model architecture built")
        print("   Input: {}".format(input_tensor.shape))
        print("   Output: {}".format(predictions.shape))

        # Load checkpoint weights
        print("\n--- Loading Weights ---")
        print("Loading from: {}".format(CHECKPOINT_FILE))
        
        saver = tf.train.Saver()
        try:
            saver.restore(sess, CHECKPOINT_FILE)
            print("✅ Weights loaded successfully")
        except Exception as e:
            print("❌ Failed to load weights: {}".format(e))
            return False

        # Test the loaded model
        if not test_model_inference(sess, input_tensor, predictions):
            print("⚠️  Model test failed, but continuing with export...")

        # Export to SavedModel
        print("\n--- Exporting SavedModel ---")
        print("Export location: {}".format(EXPORT_DIR))
        
        # Remove existing export if it exists
        if os.path.exists(EXPORT_DIR):
            import shutil
            shutil.rmtree(EXPORT_DIR)
            print("Removed existing export directory")
        
        # Create parent directories
        os.makedirs(os.path.dirname(EXPORT_DIR), exist_ok=True)
            
        try:
            tf.saved_model.simple_save(
                sess,
                EXPORT_DIR,
                inputs={'input': input_tensor},
                outputs={'predictions': predictions, 'logits': logits}
            )
            print("✅ SavedModel exported successfully!")
            
        except Exception as e:
            print("❌ Export failed: {}".format(e))
            return False

    # Final verification
    print("\n--- Export Summary ---")
    if os.path.exists(EXPORT_DIR):
        # Show directory contents
        total_size = 0
        for root, dirs, files in os.walk(EXPORT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                total_size += size
        
        print("✅ SavedModel created: {}".format(EXPORT_DIR))
        print("   Total size: {:.1f} MB".format(total_size / (1024*1024)))
        
        print("\n--- Usage Instructions ---")
        print("In modern TensorFlow:")
        print("  import tensorflow as tf")
        print("  model = tf.saved_model.load('{}')".format(EXPORT_DIR))
        print("  predictions = model.signatures['serving_default'](input=your_images)")
        
        return True
    else:
        print("❌ SavedModel export verification failed")
        return False

if __name__ == '__main__':
    success = main()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 CONVERSION COMPLETED SUCCESSFULLY!")
        print("Your model is ready for use in modern TensorFlow.")
    else:
        print("❌ CONVERSION FAILED")
        print("Please check the error messages above.")