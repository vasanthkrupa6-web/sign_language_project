import pickle
import shutil
from pathlib import Path
import tensorflow as tf

root = Path.cwd()
model_path = root / "model" / "dynamic_sign_model.keras"
labels_path = root / "model" / "dynamic_labels.pkl"
root_tflite = root / "model" / "dynamic_sign_model.tflite"
mobile_dir = root / "mobile_app" / "assets"
mobile_tflite = mobile_dir / "models" / "dynamic_sign_model.tflite"
mobile_labels = mobile_dir / "labels.txt"

print("Loading latest trained model...")
model = tf.keras.models.load_model(model_path)

with open(labels_path, "rb") as f:
    labels = pickle.load(f)

labels = [str(label) for label in labels]

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)
print("Trained labels:", labels)

if tuple(model.input_shape[1:]) != (100, 63):
    raise ValueError("Expected model input (None, 100, 63). Export stopped.")

if model.output_shape[-1] != len(labels):
    raise ValueError("Output count does not match label count. Export stopped.")

if len(labels) != 12:
    raise ValueError(f"Expected 12 trained labels, found {len(labels)}.")

if "sleep" in labels:
    raise ValueError("Unexpected sleep label. Export stopped.")

if "phone" not in labels:
    raise ValueError("Expected phone label is missing. Export stopped.")

mobile_dir.mkdir(parents=True, exist_ok=True)
mobile_tflite.parent.mkdir(parents=True, exist_ok=True)

# Back up the current mobile model and labels before replacing them.
if mobile_tflite.exists():
    backup = mobile_tflite.with_name("dynamic_sign_model_30frame_backup.tflite")
    if not backup.exists():
        shutil.copy2(mobile_tflite, backup)
        print("Old mobile model backed up to:", backup)

if mobile_labels.exists():
    backup_labels = mobile_labels.with_name("labels_backup.txt")
    if not backup_labels.exists():
        shutil.copy2(mobile_labels, backup_labels)
        print("Old labels backed up to:", backup_labels)

print("Converting latest Keras model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_bytes = converter.convert()

# Write the same newly converted model to both locations.
root_tflite.write_bytes(tflite_bytes)
mobile_tflite.write_bytes(tflite_bytes)

# Keep the mobile label order exactly the same as the trained model.
mobile_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")

# Verify the exported mobile model.
interpreter = tf.lite.Interpreter(model_path=str(mobile_tflite))
interpreter.allocate_tensors()
input_shape = interpreter.get_input_details()[0]["shape"].tolist()
output_shape = interpreter.get_output_details()[0]["shape"].tolist()

print("\nEXPORT COMPLETE")
print("Mobile input shape:", input_shape)
print("Mobile output shape:", output_shape)
print("Mobile labels:")
for i, label in enumerate(labels):
    print(f"{i}: {label}")

if input_shape != [1, 100, 63] or output_shape != [1, 12]:
    raise ValueError("Exported model shape verification failed.")
