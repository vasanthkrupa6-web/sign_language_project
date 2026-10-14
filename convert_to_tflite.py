import os
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "model/dynamic_sign_model.keras"
TFLITE_PATH = "model/dynamic_sign_model.tflite"

SEQUENCE_LENGTH = 30
FEATURES = 63


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("❌ Model not found!")
    print(f"Expected: {MODEL_PATH}")
    exit()


print("=" * 60)
print("       SIGNECHO TFLITE CONVERSION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n🔄 Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("✅ Model loaded!")

print(f"\nInput shape  : {model.input_shape}")
print(f"Output shape : {model.output_shape}")


# ============================================================
# CREATE CONCRETE FUNCTION
# ============================================================

print("\n🔄 Preparing model for conversion...")

@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=(1, SEQUENCE_LENGTH, FEATURES),
            dtype=tf.float32
        )
    ]
)
def serving_function(x):

    return model(x, training=False)


concrete_function = serving_function.get_concrete_function()


# ============================================================
# CONVERT
# ============================================================

print("🔄 Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_function],
    model
)

tflite_model = converter.convert()


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    TFLITE_PATH,
    "wb"
) as f:

    f.write(tflite_model)


# ============================================================
# VERIFY FILE
# ============================================================

file_size = os.path.getsize(
    TFLITE_PATH
)

print("\n" + "=" * 60)
print("✅ TFLITE CONVERSION SUCCESSFUL")
print("=" * 60)

print(f"Model : {TFLITE_PATH}")
print(f"Size  : {file_size / (1024 * 1024):.2f} MB")

print("=" * 60)