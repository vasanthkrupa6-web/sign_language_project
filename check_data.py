import numpy as np
import os

file_path = "ECHO-20/drink/person_01/sample_0000.npy"

data = np.load(file_path)

print("================================")
print("       ECHO-20 DATA CHECK")
print("================================")

print("File:", file_path)
print("Data type:", data.dtype)
print("Shape:", data.shape)
print("Number of values:", data.size)
print("First 10 values:", data[:10])
print("Minimum value:", np.min(data))
print("Maximum value:", np.max(data))
print("Contains NaN:", np.isnan(data).any())
print("Contains infinity:", np.isinf(data).any())

if data.shape == (63,):
    print("\n✅ SUCCESS: 63 MediaPipe features found!")
else:
    print("\n❌ ERROR: Expected 63 features.")

if not np.isnan(data).any() and not np.isinf(data).any():
    print("✅ No invalid values found!")
else:
    print("❌ Invalid values detected!")

print("================================")