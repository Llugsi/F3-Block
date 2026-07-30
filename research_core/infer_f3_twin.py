# [Script 7 & 8] Quantitative evaluation on F3 digital twin
# =====================================================================
# 1. FORMAT FIELD DATA TO PYTORCH TENSOR STRUCTURE
# =====================================================================
# Enforce explicit formatting matrix bounds: (Batch=1, Channel=1, Height, Width)
f3_tensor_input = torch.tensor(f3_input).unsqueeze(0).unsqueeze(0).to(device)

# 2. Freeze parameters and configure network into evaluation mode
model.eval()

with torch.no_grad():
    # Process the entire North Sea structural profile via single forward call
    f3_output_tensor = model(f3_tensor_input)

# [DIMENSIONAL ARTIFACT RECOVERY]: Specify explicit Batch and Channel indices (0, 0)
# Ensures f3_denoised returns to a pure 2D NumPy array structure (512, 256)
f3_denoised = f3_output_tensor[0, 0].cpu().numpy()

# =====================================================================
# 3. QUANTITATIVE ANALYSIS PRE-REPORTS (MSE STRUCTURAL GAINS)
# =====================================================================
mse_f3_noisy = np.mean((f3_input - f3_ground_truth) ** 2)
mse_f3_filtered = np.mean((f3_denoised - f3_ground_truth) ** 2)
improvement_f3 = ((mse_f3_noisy - mse_f3_filtered) / mse_f3_noisy) * 100

print("📊 PRELIMINARY PERFORMANCE EVALUATION ON DIGITAL TWIN (F3 BLOCK):")
print(f"   - Initial MSE of corrupted profile: {mse_f3_noisy:.4f}")
print(f"   - Final MSE after network inference: {mse_f3_filtered:.4f}")
print(f"   - Relative structural improvement percentage: {improvement_f3:.2f}%\n")

# =====================================================================
# 4. QUALITATIVE DOUBLE-COLUMN WORKFLOW GENERATION (3 PANELES)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

cmap_style = 'seismic'
color_bounds = {'vmin': -2.0, 'vmax': 2.0}

# Panel A: Raw input fields
axes[0].imshow(f3_input, cmap=cmap_style, aspect='auto', **color_bounds)
axes[0].set_title("A) INPUT: Raw Corrupted Profile\n(Emulated North Sea F3 Setup)", fontsize=11, fontweight='bold')
axes[0].set_ylabel("Two-Way Travel Time Samples / Depth (t)", fontsize=10)
axes[0].set_xlabel("Seismic Line Index (x)", fontsize=10)

# Panel B: Cleaned inference profile
axes[1].imshow(f3_denoised, cmap=cmap_style, aspect='auto', **color_bounds)
axes[1].set_title("B) OUTPUT: Self-Supervised Inference\n(Proposed TNNLS-BSN Framework)", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Seismic Line Index (x)", fontsize=10)

# Panel C: Hidden pristine targets
axes[2].imshow(f3_ground_truth, cmap=cmap_style, aspect='auto', **color_bounds)
axes[2].set_title("C) GROUND TRUTH: Pristine Reference\n(Hidden Structural Stratigraphy)", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Seismic Line Index (x)", fontsize=10)

for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.show()
