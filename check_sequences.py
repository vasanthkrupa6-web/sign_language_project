import numpy as np
import os

folder = "ECHO-20/drink/person_01_sequence_test"

print("=" * 50)
print("       ECHO-20 SEQUENCE CHECK")
print("=" * 50)

files = sorted([
    f for f in os.listdir(folder)
    if f.endswith(".npy")
])

print("Number of samples:", len(files))

for file in files:

    path = os.path.join(folder, file)

    data = np.load(path)

    print(f"\n{file}")
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("NaN:", np.isnan(data).any())
    print("Infinity:", np.isinf(data).any())

    if data.shape == (30, 63):
        print("✅ Valid sequence")
    else:
        print("❌ Incorrect shape")

print("\n" + "=" * 50)