import os
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "dynamic_dataset"
MODEL_PATH = "model/dynamic_sign_model.keras"
LABEL_PATH = "model/dynamic_labels.pkl"

SEQUENCE_LENGTH = 100
FEATURES = 63


# ============================================================
# LOAD DATA
# ============================================================

X = []
y = []

print("=" * 60)
print("       SIGNECHO DYNAMIC MODEL TRAINING")
print("=" * 60)

if not os.path.exists(DATASET_PATH):

    print("❌ dynamic_dataset folder not found.")
    exit()


labels = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])

print(f"Classes found: {labels}")
print()


# ============================================================
# READ SEQUENCES
# ============================================================

for label in labels:

    folder_path = os.path.join(
        DATASET_PATH,
        label
    )

    files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith(".npy")
    ]

    print(
        f"Loading {label}: {len(files)} samples"
    )

    for file in files:

        file_path = os.path.join(
            folder_path,
            file
        )

        try:

            sequence = np.load(
                file_path
            )

            # ------------------------------------------------
            # CHECK SHAPE
            # ------------------------------------------------

            if sequence.shape != (
                SEQUENCE_LENGTH,
                FEATURES
            ):

                print(
                    f"⚠️ Skipping {file}"
                    f" | Shape: {sequence.shape}"
                )

                continue


            # ------------------------------------------------
            # CHECK INVALID VALUES
            # ------------------------------------------------

            if not np.all(
                np.isfinite(sequence)
            ):

                print(
                    f"⚠️ Skipping {file}"
                    f" | Invalid values"
                )

                continue


            X.append(sequence)
            y.append(label)


        except Exception as e:

            print(
                f"❌ Error loading {file}: {e}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)


print()
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(
    f"Total sequences : {len(X)}"
)

print(
    f"Sequence shape  : {X.shape}"
)


for label in labels:

    count = np.sum(
        y == label
    )

    print(
        f"{label:15s}: {count} sequences"
    )


# ============================================================
# VALIDATE DATASET
# ============================================================

if len(X) == 0:

    print("\n❌ No valid sequences found.")
    exit()


if len(labels) < 2:

    print(
        "\n⚠️ WARNING:"
    )

    print(
        "Only one class exists."
    )

    print(
        "For a meaningful classifier,"
        " we need at least TWO classes."
    )

    print(
        "\nFor now we will stop here."
    )

    exit()


# ============================================================
# ENCODE LABELS
# ============================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)


print()
print("Encoded classes:")

for index, label in enumerate(
    encoder.classes_
):

    print(
        f"{index} → {label}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print()
print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    f"Training sequences : {len(X_train)}"
)

print(
    f"Testing sequences  : {len(X_test)}"
)


# ============================================================
# BUILD LSTM
# ============================================================

print()
print("🧠 Building LSTM model...")


model = Sequential([

    LSTM(
        64,
        input_shape=(
            SEQUENCE_LENGTH,
            FEATURES
        ),
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        32
    ),

    Dropout(0.2),

    Dense(
        32,
        activation="relu"
    ),

    Dense(
        len(labels),
        activation="softmax"
    )

])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 60)
print("TRAINING LSTM")
print("=" * 60)


history = model.fit(

    X_train,
    y_train,

    validation_split=0.2,

    epochs=50,

    batch_size=16,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# EVALUATE
# ============================================================

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)


y_probability = model.predict(
    X_test
)

y_pred = np.argmax(
    y_probability,
    axis=1
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


print()
print("Classification Report:")


print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)


model.save(
    MODEL_PATH
)


with open(
    LABEL_PATH,
    "wb"
) as f:

    pickle.dump(
        encoder.classes_.tolist(),
        f
    )


print()
print("=" * 60)
print("✅ DYNAMIC MODEL SAVED")
print("=" * 60)

print(
    f"Model : {MODEL_PATH}"
)

print(
    f"Labels: {LABEL_PATH}"
)

print("=" * 60)