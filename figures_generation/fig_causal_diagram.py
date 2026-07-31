# [Script 15] 5x5 Grid vector mapping for Blind-Spot visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Compact layout configuration 
plt.rcParams['font.family'] = 'sans-serif'
fig, ax = plt.subplots(figsize=(6, 6), dpi=300) # Balanced proportions for single-column embedding

# 1. Define compact grid sizing (5x5) and its respective coordinate center
grid_size = 5
center = grid_size // 2

# Draw subtle, sharp background grid lines
for x in range(grid_size + 1):
    ax.axhline(x, color='#D9DDE2', linestyle='-', linewidth=1.5, zorder=1)
    ax.axvline(x, color='#D9DDE2', linestyle='-', linewidth=1.5, zorder=1)

# 2. Paint asymmetric 1-D directional causal convolutional streams (Subtle tones)
ax.add_patch(patches.Rectangle((center, center+1), 1, 2, facecolor='#E6F2FF', alpha=0.7, zorder=2)) # North (Blue)
ax.add_patch(patches.Rectangle((center, 0), 1, center, facecolor='#FFE6E6', alpha=0.7, zorder=2))    # South (Red)
ax.add_patch(patches.Rectangle((center+1, center), 2, 1, facecolor='#E6FFE6', alpha=0.7, zorder=2)) # East (Green)
ax.add_patch(patches.Rectangle((0, center), center, 1, facecolor='#FFF2E6', alpha=0.7, zorder=2))    # West (Orange)

# 3. Highlight the Central Pixel Anomaly (Strict Causal Blind-Spot)
blind_spot = patches.Rectangle((center, center), 1, 1, facecolor='#343A40', edgecolor='#111111', lw=3, zorder=3)
ax.add_patch(blind_spot)
ax.text(center + 0.5, center + 0.5, 'Blind\nSpot\n(y,x)', color='white', ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

# =====================================================================
# 4. CAUSAL VECTOR ANNOTATIONS 
# =====================================================================

# North Branch: Information flows strictly from top to center (y' < y)
ax.annotate('', xy=(center + 0.5, center + 1.0), xytext=(center + 0.5, grid_size - 0.1),
            arrowprops=dict(facecolor='#0066CC', edgecolor='#004499', shrink=0.01, width=4, headwidth=10), zorder=5)
ax.text(center + 0.5, grid_size + 0.15, 'North Stream\n(1-D V Causal)', color='#0066CC', ha='center', va='bottom', fontsize=12, fontweight='bold')

# South Branch: Information flows strictly from bottom to center (y' > y)
ax.annotate('', xy=(center + 0.5, center), xytext=(center + 0.5, 0.1),
            arrowprops=dict(facecolor='#CC0000', edgecolor='#990000', shrink=0.01, width=4, headwidth=10), zorder=5)
ax.text(center + 0.5, -0.15, 'South Stream\n(1-D V Causal)', color='#CC0000', ha='center', va='top', fontsize=12, fontweight='bold')

# East Branch: Information flows strictly from right to center (x' > x)
ax.annotate('', xy=(center + 1.0, center + 0.5), xytext=(grid_size - 0.1, center + 0.5),
            arrowprops=dict(facecolor='#009933', edgecolor='#006622', shrink=0.01, width=4, headwidth=10), zorder=5)
ax.text(grid_size + 0.15, center + 0.5, 'East Stream\n(1-D H Causal)', color='#009933', ha='left', va='center', fontsize=12, fontweight='bold')

# West Branch: Information flows strictly from left to center (x' < x)
ax.annotate('', xy=(center, center + 0.5), xytext=(0.1, center + 0.5),
            arrowprops=dict(facecolor='#FF8000', edgecolor='#CC6600', shrink=0.01, width=4, headwidth=10), zorder=5)
ax.text(-0.15, center + 0.5, 'West Stream\n(1-D H Causal)', color='#FF8000', ha='right', va='center', fontsize=12, fontweight="bold")

# Optimize bounds to strip out marginal dead spaces and maintain layout symmetry
ax.set_xlim(-1.6, grid_size + 1.6)
ax.set_ylim(-0.8, grid_size + 0.8)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig('blind_spot_network_diagram.png', bbox_inches='tight', dpi=300)
plt.show()
print("Genuinely synchronized architectural diagram saved as 'blind_spot_network_diagram.png'.")
