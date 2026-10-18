import pickle
from pathlib import Path
import tensorflow as tf

root = Path.cwd()
model = tf.keras.models.load_model(
    root / "model" / "dynamic_sign_model.keras"
)

with open(root / "model" / "dynamic_labels.pkl", "rb") as f:
    labels = [str(x) for x in pickle.load(f)]

assert tuple(model.input_shape[1:]) == (100, 63), model.input_shape
assert model.output_shape[-1] == len(labels), model.output_shape
assert len(labels) == 12
assert "phone" in labels and "sleep" not in labels

print("TensorFlow version:", tf.__version__)
print("Preparing concrete inference function...")

@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[1, 100, 63], dtype=tf.float32)
    ]
)
def serving_fn(x):
    return model(x, training=False)

concrete_fn = serving_fn.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_fn],
    model
)

print("Converting model...")
converted = converter.convert()

# Verify the conversion in memory before touching existing files.
interpreter = tf.lite.Interpreter(model_content=converted)
interpreter.allocate_tensors()

input_shape = interpreter.get_input_details()[0]["shape"].tolist()
output_shape = interpreter.get_output_details()[0]["shape"].tolist()

print("Converted input:", input_shape)
print("Converted output:", output_shape)

if input_shape != [1, 100, 63] or output_shape != [1, 12]:
    raise RuntimeError("Converted model has unexpected shapes.")

# Save to a temporary file first.
mobile = root / "mobile_app" / "assets" / "models" / "dynamic_sign_model.tflite"
temp = mobile.with_name("dynamic_sign_model_new.tflite")
temp.write_bytes(converted)

# Verify the temporary file from disk.
check = tf.lite.Interpreter(model_path=str(temp))
check.allocate_tensors()
assert check.get_input_details()[0]["shape"].tolist() == [1, 100, 63]
assert check.get_output_details()[0]["shape"].tolist() == [1, 12]

# Only replace the mobile model after successful verification.
temp.replace(mobile)

labels_path = root / "mobile_app" / "assets" / "labels.txt"
labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

print("\nSUCCESS: Mobile model and labels updated.")
print("Input shape:", [1, 100, 63])
print("Output shape:", [1, 12])
print("Labels:", labels)
