import os
import pickle
from sklearn.preprocessing import LabelEncoder


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "dynamic_dataset"
LABEL_PATH = "model/dynamic_labels.pkl"


# ============================================================
# FIND DATASET CLASSES
# ============================================================

labels = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(
        os.path.join(DATASET_PATH, folder)
    )
])


# ============================================================
# CREATE LABEL ENCODER
# ============================================================

encoder = LabelEncoder()

encoder.fit(labels)


# ============================================================
# SAVE LABELS
# ============================================================

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    LABEL_PATH,
    "wb"
) as f:

    pickle.dump(
        encoder.classes_.tolist(),
        f
    )


# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("       SIGNECHO LABEL FIX")
print("=" * 60)

print("\nClasses found:")

for index, label in enumerate(
    encoder.classes_
):

    print(
        f"{index:2d} → {label}"
    )


print("\nTotal classes:")
print(len(encoder.classes_))

print(
    f"\n✅ Labels saved to:"
)

print(LABEL_PATH)

print("=" * 60)