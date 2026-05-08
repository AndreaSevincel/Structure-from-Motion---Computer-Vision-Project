import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

files = [
    "cloud_000_002.npy",
    "cloud_002_004.npy",
    "cloud_004_006.npy",
    "cloud_006_008.npy",
]

clouds = [np.load(f) for f in files]
merged = np.vstack(clouds)
print(f"Total points: {len(merged)}")

np.save("merged_cloud.npy", merged)

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(merged[:, 0], merged[:, 1], merged[:, 2], s=1, c="crimson", marker=".")
ax.set_title("Merged Sparse 3D Point Cloud")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.tight_layout()
plt.savefig("merged_cloud.png", dpi=150, bbox_inches="tight")
print("Saved merged_cloud.png")