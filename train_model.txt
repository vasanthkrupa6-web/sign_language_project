import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("📂 Loading dataset...\n")

DATA_DIR = "dataset"
X, y = [], []

for label in os.listdir(DATA_DIR):
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        continue
    files = os.listdir(folder)
    print(f"  ✅ '{label}' — {len(files)} samples")
    for file in files:
        if file.endswith(".npy"):
            data = np.load(os.path.join(folder, file))
            if len(data) == 63:
                X.append(data)
                y.append(label)

print(f"\n📊 Total samples: {len(X)}")
print(f"🏷️  Classes: {list(set(y))}\n")

if len(set(y)) < 2:
    print("❌ Need at least 2 signs to train. Collect more data first!")
    exit()

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🧠 Training model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
with open("model/sign_model.pkl", "wb") as f:
    pickle.dump({"model": model, "labels": list(set(y))}, f)

print("💾 Model saved to model/sign_model.pkl")