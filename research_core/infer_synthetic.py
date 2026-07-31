# [Script 5] Global synthetic inference pipeline and 3-panel metrics evaluator
num_traces = 256
num_samples = 512
t = np.linspace(0, 10, num_samples)
x = np.arange(num_traces)

# Deploy generator with updated 2D spatial stochastic noise structures
n1_global, n2_global, clean_global = seismic_data_generation(
    t, x, peak_1=1.4, peak_2=1.2, peak_3=1.0, axis=3.0, 
    height=0.3, traces=128, m=0.003, b=4.2, x_fault=128, throw=0.18
)

# =====================================================================
# 2. PYTORCH INFERENCE SETUP (EVALUATION MODE)
# =====================================================================
input_tensor = torch.from_numpy(n1_global.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

model.eval() 

with torch.no_grad():
    reconstructed_tensor = model(input_tensor)

denoised_data = reconstructed_tensor.squeeze().cpu().numpy()

# =====================================================================
# 3. QUANTITATIVE PERFORMANCE SUMMARY
# =====================================================================
data_range = clean_global.max() - clean_global.min()

mse_noisy = np.mean((n1_global - clean_global) ** 2)
mse_clean = np.mean((denoised_data - clean_global) ** 2)
mse_improvement = ((mse_noisy - mse_clean) / mse_noisy) * 100

psnr_noisy = psnr(clean_global, n1_global, data_range=data_range)
psnr_clean = psnr(clean_global, denoised_data, data_range=data_range)

ssim_noisy = ssim(clean_global, n1_global, data_range=data_range)
ssim_clean = ssim(clean_global, denoised_data, data_range=data_range)

print(f"--- GLOBAL QUANTITATIVE EVALUATION REPORT: 7-LAYER PROFILE (IEEE TNNLS) ---")
print(f"   [MSE Metric] Initial: {mse_noisy:.4f}  -->  Final: {mse_clean:.4f}  (Improvement: {mse_improvement:.2f}%)")
print(f"   [PSNR Metric] Initial: {psnr_noisy:.2f} dB -->  Final: {psnr_clean:.2f} dB (Net Gain: +{psnr_clean - psnr_noisy:.2f} dB)")
print(f"   [SSIM Metric] Initial: {ssim_noisy:.4f}  -->  Final: {ssim_clean:.4f}  (Net Gain: +{ssim_clean - ssim_noisy:.4f})")
print(f"----------------------------------------------------------------\n")

# =====================================================================
# 4. SCIENTIFIC VISUALIZATION CODES (3-PANEL COMPARISON WORKFLOW)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

cmap_style = 'seismic'
color_bounds = {'vmin': -2.0, 'vmax': 2.0}

# Panel A: Corrupted Input
axes[0].imshow(n1_global, cmap=cmap_style, aspect='auto', **color_bounds)
axes[0].set_title(f"A) INPUT: 7 Harmonic Corrupted Layers\nPSNR: {psnr_noisy:.2f} dB | SSIM: {ssim_noisy:.4f}", fontsize=11, fontweight='bold')
axes[0].set_xlabel("Seismic Trace Index (x)", fontsize=10)
axes[0].set_ylabel("Time Travel Samples / Depth (t)", fontsize=10)

# Panel B: Reconstructed Output
axes[1].imshow(denoised_data, cmap=cmap_style, aspect='auto', **color_bounds)
axes[1].set_title(f"B) OUTPUT: Reconstructed Filtered BSN\nPSNR: {psnr_clean:.2f} dB | SSIM: {ssim_clean:.4f}", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Seismic Trace Index (x)", fontsize=10)

# Panel C: Ground Truth Target
axes[2].imshow(clean_global, cmap=cmap_style, aspect='auto', **color_bounds)
axes[2].set_title("C) GROUND TRUTH: Isolated Reflectors\n(Central Fault Discontinuity at Trace 128)", fontsize=11, fontweight='bold')
axes[2].set_xlabel("Seismic Trace Index (x)", fontsize=10)

for ax in axes:
    ax.grid(False)

plt.tight_layout()
plt.show()
