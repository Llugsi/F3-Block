# [Script 12] Dual-axis PSNR/SSIM high-density bar-chart generator
import matplotlib.pyplot as plt
import numpy as np

# Updated empirical data extracted directly from your active memory evaluation
methods = [
    "Raw Input\n(Corrupted)", 
    "Classical Filter\n(f-x Deconv.)", 
    "Baseline UNet\n(Data Leakage)", 
    "Proposed Network\n(BSN)"
]
psnr_values = [14.31, 17.54, 32.81, 16.22]
ssim_values = [0.0525, 0.1634, 0.8166, 0.0804]

# Canvas optimization setup for IEEE double-column grid layout
fig, ax1 = plt.subplots(figsize=(10, 6.5), dpi=300)

# --- LEFT AXIS CONFIGURATION: PSNR (High-Density Bar Layout) ---
bar_colors = ['#95a5a6', '#34495e', '#e74c3c', '#2ecc71'] # Professional color coding for journal
bars = ax1.bar(methods, psnr_values, color=bar_colors, width=0.45, edgecolor='black', lw=1.5, alpha=0.85)

ax1.set_ylabel("Peak Signal-to-Noise Ratio (PSNR in dB)", fontsize=14, fontweight='bold', color='black', labelpad=12)
ax1.set_ylim(0, 42) # Extended vertical space for text annotations
ax1.tick_params(axis='y', labelcolor='black', labelsize=13, width=1.5, length=6)
ax1.tick_params(axis='x', labelsize=13)
ax1.grid(True, axis='y', linestyle='--', alpha=0.4, zorder=0)

# Add clear high-visibility values on top of each metric bar
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f} dB',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5),  # Vertical offset of 5 points for clarity
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

# --- RIGHT AXIS CONFIGURATION: SSIM (Overlay Line Graph) ---
ax2 = ax1.twinx()  
line_color = '#2980b9' # Strong contrasting professional blue for the SSIM trend line
line = ax2.plot(methods, ssim_values, color=line_color, marker='o', linewidth=3.0, 
                markersize=10, linestyle='-', label='SSIM Index', zorder=5)

ax2.set_ylabel("Structural Similarity Index (SSIM)", fontsize=14, fontweight='bold', color=line_color, labelpad=12)
ax2.set_ylim(-0.05, 1.15) # Standardized bounded axis for structural metrics
ax2.tick_params(axis='y', labelcolor=line_color, labelsize=13, width=1.5, length=6)

# Optimized placement of SSIM floating annotations to avoid horizontal overlapping
for i, txt in enumerate(ssim_values):
    # Dynamic vertical offset calculation based on model type to maximize visual space
    y_offset = 10 if methods[i] != "Proposed Red\n(TNNLS-BSN)" else -18
    x_offset = 0 if methods[i] != "Proposed Red\n(TNNLS-BSN)" else -10
    
    ax2.annotate(f'{txt:.4f}', (methods[i], ssim_values[i]), 
                 xytext=(x_offset, y_offset), textcoords='offset points', 
                 color=line_color, fontsize=12, fontweight='bold', ha='center')

# --- TITLES AND HIGH-RESOLUTION EXPORT CONTROLS ---
plt.title("Quantitative Performance Comparison Trends: North Sea F3 Digital Twin", fontsize=15, fontweight='bold', pad=20)
fig.tight_layout()

# Save into format matching your LaTeX figure handle 'fig:benchmark_metrics'
plt.savefig('benchmark_metrics.jpg', bbox_inches='tight', dpi=300)
plt.show()
print("✅ Quantitative trend figure compiled and saved as 'benchmark_metrics.jpg'.")
