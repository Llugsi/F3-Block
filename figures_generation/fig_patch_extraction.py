# [Script 17] Multi-index PyTorch DataLoader online stochastic sampler diagram
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# High-density layout configuration for IEEE double-column template compatibility
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300) 

# =====================================================================
# 1. FULL SEISMIC PROFILE REPRESENTATION MATRIX X
# =====================================================================
large_matrix = patches.Rectangle((0.5, 0.5), 4.5, 4.0, facecolor="#F8F9FA", edgecolor="#222222", lw=3.5)
ax.add_patch(large_matrix)

x_lines = np.linspace(0.5, 5.0, 15)
for xl in x_lines:
    ax.axvline(xl, ymin=0.12, ymax=0.88, color="#DDE1E5", lw=1.0, zorder=1)

ax.text(2.75, 4.65, r"Full Seismic Profile $\mathbf{X} \in \mathbb{R}^{T \times X}$", ha="center", va="bottom", fontsize=18, fontweight="bold")
ax.text(-0.05, 2.5, "Time Axis (T)", ha="center", va="center", rotation=90, fontsize=16, color="#111111", fontweight="bold")
ax.text(2.75, -0.05, "Spatial Traces (X)", ha="center", va="top", fontsize=16, color="#111111", fontweight="bold")

# =====================================================================
# 2. ONLINE RANDOM PATCHES SELECTION
# =====================================================================
# Patch 1: North/Upper Sector (Blue Mapping)
p1 = patches.Rectangle((0.8, 3.2), 0.8, 0.8, facecolor="#0066CC", alpha=0.25, edgecolor="#0066CC", lw=3.5, ls="--", zorder=3)
ax.add_patch(p1)
ax.text(1.2, 3.6, r"$\mathbf{(y_1, x_1)}$", ha="center", va="center", fontsize=16, color="#004499", fontweight="bold")

# Patch 2: Central Sector (Red Mapping)
p2 = patches.Rectangle((3.2, 1.8), 0.8, 0.8, facecolor="#CC0000", alpha=0.25, edgecolor="#CC0000", lw=3.5, ls="--", zorder=3)
ax.add_patch(p2)
ax.text(3.6, 2.2, r"$\mathbf{(y_2, x_2)}$", ha="center", va="center", fontsize=16, color="#990000", fontweight="bold")

# Patch 3: Lower Sector (Green Mapping)
p3 = patches.Rectangle((1.5, 0.8), 0.8, 0.8, facecolor="#009933", alpha=0.25, edgecolor="#009933", lw=3.5, ls="--", zorder=3)
ax.add_patch(p3)
ax.text(1.9, 1.2, r"$\mathbf{(y_3, x_3)}$", ha="center", va="center", fontsize=16, color="#006622", fontweight="bold")

# =====================================================================
# 3. PYTORCH DATALOADER BATCH MULTI-INDEXING CONTAINER
# =====================================================================
dataloader_box = patches.Rectangle((8.4, 0.8), 2.2, 3.2, facecolor="#FFF5EC", edgecolor="#E65C00", lw=3.5, zorder=2)
ax.add_patch(dataloader_box)
ax.text(9.5, 4.15, "PyTorch DataLoader\n" + r"Tensor: $[B, 1, H_p, W_p]$", ha="center", va="bottom", fontsize=15, fontweight="bold", color="#A64200")

for i in range(3):
    offset = i * 0.22
    batch_patch = patches.Rectangle((8.6 + offset, 1.2 + offset), 0.8, 0.8, 
                                    facecolor="#FFFFFF", edgecolor="#FF8000", lw=2.5, alpha=1.0, zorder=4+i)
    ax.add_patch(batch_patch)
    ax.text(9.0 + offset, 1.6 + offset, r"$\mathbf{P}_{%d}$" % (i+1), ha="center", va="center", fontsize=14, fontweight="bold", color="#FF8000", zorder=7+i)

# =====================================================================
# 4. PATH CURVATURE ROUTINGS
# =====================================================================
ax.text(6.4, 2.4, "Online Stochastic\nSampling", ha="center", va="center", fontsize=16, color="#111111", fontweight="bold")
ax.text(6.4, 1.5, r"$(y, x) \sim \mathcal{U}(0, \cdot)$ per Epoch", ha="center", va="center", fontsize=15, color="#333333", fontweight="bold")

ax.annotate('', xy=(8.4, 3.2), xytext=(1.6, 3.6), arrowprops=dict(arrowstyle="-|>", color="#0066CC", lw=3.0, connectionstyle="arc3,rad=-0.22", mutation_scale=20), zorder=8)
ax.annotate('', xy=(8.4, 2.4), xytext=(4.0, 2.2), arrowprops=dict(arrowstyle="-|>", color="#CC0000", lw=3.0, connectionstyle="arc3,rad=0.08", mutation_scale=20), zorder=8)
ax.annotate('', xy=(8.4, 1.6), xytext=(2.3, 1.2), arrowprops=dict(arrowstyle="-|>", color="#009933", lw=3.0, connectionstyle="arc3,rad=0.25", mutation_scale=20), zorder=8)

ax.set_xlim(-0.4, 11.4)
ax.set_ylim(-0.4, 5.5)
ax.axis('off')

plt.tight_layout()
plt.savefig('seismic_patch_extraction.png', bbox_inches='tight', dpi=300)
plt.show()
print("✅ High-visibility patch extraction diagram compiled and saved as 'seismic_patch_extraction.png'.")
