# [Script 16] Hardware bus-corrected attention gate schematic
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Canvas configuration optimized for massive visibility in IEEE templates
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300) # Slightly taller grid for clean bus spacing

# =====================================================================
# 1. RIGID COORDINATE HARDWARE PATCH MAPPING (HIGH-VISIBILITY BOUNDS)
# =====================================================================

# Cardinal directional input arrays and standalone stream-isolated Conv2D layers
y_positions = [4.5, 3.2, 1.9, 0.6]
labels_inputs = [r"$\mathbf{F}_{\mathcal{N}}$", r"$\mathbf{F}_{\mathcal{S}}$", r"$\mathbf{F}_{\mathcal{E}}$", r"$\mathbf{F}_{\mathcal{W}}$"]

for y, l_in in zip(y_positions, labels_inputs):
    # Isolated Feature Array Box (F_dir) - MASSIVE FONT
    box_in = patches.FancyBboxPatch((0.0, y - 0.28), 0.75, 0.56, boxstyle="round,pad=0.03",
                                    facecolor="#F8F9FA", edgecolor="#495057", lw=4.0, zorder=3)
    ax.add_patch(box_in)
    ax.text(0.375, y, l_in, ha="center", va="center", fontsize=26, fontweight="bold", zorder=4)
    
    # Standalone Stream-Isolated 1x1 Conv Box (Thick structural styling)
    box_conv1x1 = patches.Rectangle((1.6, y - 0.28), 1.7, 0.56, facecolor="#E6F2FF", edgecolor="#0066CC", lw=4.0, zorder=3)
    ax.add_patch(box_conv1x1)
    ax.text(2.45, y, r"$\mathbf{Conv2D_{1 \times 1}}$", ha="center", va="center", fontsize=14, fontweight="bold", zorder=4)
    
    # Thick internal directional streaming arrow
    ax.annotate('', xy=(1.6, y), xytext=(0.87, y),
                arrowprops=dict(arrowstyle="-|>", color="#495057", lw=3.5, mutation_scale=18), zorder=4)

# Score Concatenation Block Box - HIGH RESOLUTION BOUNDS
box_concat = patches.Rectangle((4.0, 0.28), 1.9, 4.8, facecolor="#E9ECEF", edgecolor="#343A40", lw=4.0, zorder=2)
ax.add_patch(box_concat)
box_concat_text = "Score\nConcatenation\nBlock\n" + r"$\mathbf{S}_{\mathbf{cat}}$"
ax.text(4.95, 2.68, box_concat_text, ha="center", va="center", fontsize=18, fontweight="bold", zorder=3)

# Spatial Softmax Probability Field Mapping Box - FONT UPGRADE
box_softmax = patches.Rectangle((6.6, 2.0), 2.8, 1.3, facecolor="#FFE6E6", edgecolor="#CC0000", lw=4.0, zorder=2)
ax.add_patch(box_softmax)
box_softmax_text = "Spatial Softmax\n" + r"$\mathbf{[W}_{\mathcal{N}}\mathbf{, \dots, W}_{\mathcal{W}}\mathbf{]}$"
ax.text(8.0, 2.65, box_softmax_text, ha="center", va="center", fontsize=18, fontweight="bold", zorder=3)

# Weighted Aggregation Operator Circle - GIANT SYMBOLS
circle_op = patches.Circle((10.7, 2.65), 0.5, facecolor="#E6FFE6", edgecolor="#009933", lw=4.0, zorder=2)
ax.add_patch(circle_op)
ax.text(10.7, 2.65, r"$\mathbf{\sum \odot}$", ha="center", va="center", fontsize=28, fontweight="bold", zorder=3)
ax.text(10.7, 1.8, "Weighted\nAggregation", ha="center", va="top", fontsize=14, color="#006622", fontweight="bold")

# Output Grid Target Box: Fused Tensor (F_fused) - FONT UPGRADE
box_output = patches.FancyBboxPatch((12.1, 2.0), 2.4, 1.3, boxstyle="round,pad=0.03",
                                     facecolor="#FFF2E6", edgecolor="#FF8000", lw=4.0, zorder=2)
ax.add_patch(box_output)
box_output_text = "Fused Tensor\n" + r"$\mathbf{F}_{\mathbf{fused}}$"
ax.text(13.3, 2.65, box_output_text, ha="center", va="center", fontsize=18, fontweight="bold", zorder=3)


# =====================================================================
# 2. VECTOR STREAM ROUTING AND BUS ENFORCEMENT (THICKER ARROWS)
# =====================================================================

# Route independent convolutional branch scores into Concat Block
targets_concat_y = [3.9, 3.0, 2.2, 1.3]
for y_in, y_tar in zip(y_positions, targets_concat_y):
    ax.plot([3.3, 3.7, 3.7], [y_in, y_in, y_tar], color="#0066CC", lw=3.0, zorder=1)
    ax.annotate('', xy=(4.0, y_tar), xytext=(3.7, y_tar),
                arrowprops=dict(arrowstyle="-|>", color="#0066CC", lw=3.0, mutation_scale=18), zorder=4)

# Causal connection: From Concatenation Block to Spatial Softmax
ax.annotate('', xy=(6.6, 2.65), xytext=(5.9, 2.65), 
            arrowprops=dict(arrowstyle="-|>", color="#343A40", lw=3.5, mutation_scale=22), zorder=4)

# Probability distribution connection: From Softmax to Weighted Aggregation Circle
ax.annotate('', xy=(10.2, 2.65), xytext=(9.4, 2.65), 
            arrowprops=dict(arrowstyle="-|>", color="#CC0000", lw=3.5, mutation_scale=22), zorder=4)

# Integrated tensor mapping: From Aggregation Operator to Fused Output Tensor
ax.annotate('', xy=(12.1, 2.65), xytext=(11.2, 2.65), 
            arrowprops=dict(arrowstyle="-|>", color="#009933", lw=3.5, mutation_scale=22), zorder=4)

# =====================================================================
# ARTIFACT CORRECTION
# =====================================================================
# 1. Draw short horizontal connections from each box to the master bus (X=-0.25)
for y in y_positions:
    ax.plot([0.0, -0.25], [y, y], color="#495057", linestyle="-", lw=2.5, zorder=1)

# 2. Plot a SINGLE, continuous vertical main bus on the left side
ax.plot([-0.25, -0.25], [0.6, 5.7], color="#495057", linestyle="-", lw=2.5, zorder=1)

# 3. Connect the continuous upper horizontal line to the circular operator (Y=5.7)
ax.plot([-0.25, 10.7], [5.7, 5.7], color="#495057", linestyle="-", lw=2.5, zorder=1)

# 4. A single, clean downward arrow that injects data into the circle without overlapping
ax.annotate('', xy=(10.7, 3.15), xytext=(10.7, 5.7), 
            arrowprops=dict(arrowstyle="-|>", color="#495057", lw=3.0, mutation_scale=22), zorder=4)

# Overhead main label routing - INCREASED VISIBILITY
skip_label = r"Hermetic Feature Preservation Bus ($\mathbf{F}_{\mathbf{D}}$)"
ax.text(5.0, 6.1, skip_label, ha="center", va="center", fontsize=18, color="#212529", fontweight="bold")

# Structural bounding constraints to prevent text cutting
ax.set_xlim(-0.8, 15.0)
ax.set_ylim(0.0, 6.7)
ax.axis('off')

plt.tight_layout()
plt.savefig('blind_spot_attention_gate_hermetic.png', bbox_inches='tight', dpi=300)
plt.show()
print("Fully-corrected attention gate schematic saved as 'blind_spot_attention_gate_hermetic.png'.")
