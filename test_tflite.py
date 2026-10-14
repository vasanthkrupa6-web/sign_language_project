import os
import pickle
import numpy as np
import tensorflow as tf


# ============================================================
# SETTINGS
# ============================================================

KERAS_MODEL_PATH = "model/dynamic_sign_model.keras"
TFLITE_MODEL_PATH = "model/dynamic_sign_model.tflite"
LABEL_PATH = "model/dynamic_labels.pkl"

DATASET_PATH = "dynamic_dataset"

SEQUENCE_LENGTH = 30
FEATURES = 63


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 60)
print("       SIGNECHO TFLITE MODEL TEST")
print("=" * 60)

with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)

print("\nLabels:")
print(labels)


# ============================================================
# LOAD KERAS MODEL
# ============================================================

print("\n🔄 Loading Keras model...")

keras_model = tf.keras.models.load_model(
    KERAS_MODEL_PATH,
    compile=False
)

print("✅ Keras model loaded")


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("\n🔄 Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded")


print("\nTFLite input:")
print(input_details[0]["shape"])
print(input_details[0]["dtype"])

print("\nTFLite output:")
print(output_details[0]["shape"])
print(output_details[0]["dtype"])


# ============================================================
# FIND ONE TEST SAMPLE
# ============================================================

test_sequence = None
test_label = None

for label in sorted(os.listdir(DATASET_PATH)):

    folder = os.path.join(
        DATASET_PATH,
        label
    )

    if not os.path.isdir(folder):
        continue

    files = [
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ]

    if len(files) > 0:

        file_path = os.path.join(
            folder,
            files[0]
        )

        sequence = np.load(
            file_path
        )

        if sequence.shape == (
            SEQUENCE_LENGTH,
            FEATURES
        ):

            test_sequence = sequence
            test_label = label
            break


# ============================================================
# CHECK SAMPLE
# ============================================================

if test_sequence is None:

    print("\n❌ No valid test sequence found.")
    exit()


print("\n" + "=" * 60)
print("TEST SAMPLE")
print("=" * 60)

print(f"Actual sign : {test_label}")
print(f"Shape       : {test_sequence.shape}")


# ============================================================
# PREPARE INPUT
# ============================================================

input_data = np.expand_dims(
    test_sequence,
    axis=0
).astype(np.float32)


# ============================================================
# KERAS PREDICTION
# ============================================================

print("\n🔮 Keras prediction...")

keras_probability = keras_model.predict(
    input_data,
    verbose=0
)[0]

keras_index = np.argmax(
    keras_probability
)

keras_prediction = labels[
    keras_index
]

keras_confidence = keras_probability[
    keras_index
]


# ============================================================
# TFLITE PREDICTION
# ============================================================

print("🔮 TFLite prediction...")

interpreter.set_tensor(
    input_details[0]["index"],
    input_data
)

interpreter.invoke()

tflite_probability = interpreter.get_tensor(
    output_details[0]["index"]
)[0]

tflite_index = np.argmax(
    tflite_probability
)

tflite_prediction = labels[
    tflite_index
]

tflite_confidence = tflite_probability[
    tflite_index
]


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPARISON")
print("=" * 60)

print(f"\nActual sign:")
print(f"  {test_label}")

print(f"\nKeras:")
print(
    f"  Prediction : {keras_prediction}"
)

print(
    f"  Confidence : "
    f"{keras_confidence * 100:.2f}%"
)

print(f"\nTFLite:")
print(
    f"  Prediction : {tflite_prediction}"
)

print(
    f"  Confidence : "
    f"{tflite_confidence * 100:.2f}%"
)


# ============================================================
# COMPARE
# ============================================================

print("\n" + "=" * 60)

if keras_prediction == tflite_prediction:

    print("✅ PREDICTIONS MATCH!")

else:

    print("⚠️ PREDICTIONS DO NOT MATCH!")


difference = np.max(
    np.abs(
        keras_probability -
        tflite_probability
    )
)

print(
    f"\nMaximum probability difference:"
    f" {difference:.8f}"
)

print("=" * 60)