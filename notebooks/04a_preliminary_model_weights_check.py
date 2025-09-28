import tensorflow as tf
import numpy as np

# Path to the SavedModel you exported
EXPORT_DIR = "models/fid_classification/tf1_from_community_allow_train/1"

print("TensorFlow version:", tf.__version__)

# Load SavedModel
model = tf.saved_model.load(EXPORT_DIR)
print("✅ Model loaded successfully")

# The default serving signature
infer = model.signatures["serving_default"]

# Create dummy input (match training size: 299x299x3)
dummy_input = np.random.rand(1, 299, 299, 3).astype(np.float32)

# Run inference
outputs = infer(input=tf.convert_to_tensor(dummy_input))

print("\n--- Inference Output ---")
for key, value in outputs.items():
    print(f"{key}: shape={value.shape}")

# Example: predictions vector
preds = outputs["predictions"].numpy()[0]
print("Sum of probs:", preds.sum())  # should be ~1.0
print("Top class:", preds.argmax(), "with confidence", preds.max())
