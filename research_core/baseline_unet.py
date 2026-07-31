# [Script 9 & 14B] Standard UNet baseline architecture with bilinear interpolation
# =====================================================================
# 3. SCIENTIFIC METRICS CONSOLIDATION MODULE 
# =====================================================================
# Compute true dynamic range from pristine structural benchmarks
data_range_f3 = f3_ground_truth.max() - f3_ground_truth.min()

# 1. Structural MSE Calculations
mse_f3_noisy = np.mean((f3_input - f3_ground_truth) ** 2)
mse_f3_filtered = np.mean((f3_denoised - f3_ground_truth) ** 2)
improvement_f3_mse = ((mse_f3_noisy - mse_f3_filtered) / mse_f3_noisy) * 100

# 2. Peak Signal-to-Noise Ratio (PSNR Tracking via skimage backend)
psnr_f3_noisy = psnr(f3_ground_truth, f3_input, data_range=data_range_f3)
psnr_f3_filtered = psnr(f3_ground_truth, f3_denoised, data_range=data_range_f3)
gain_f3_psnr = psnr_f3_filtered - psnr_f3_noisy

# 3. Structural Similarity Index Metric (SSIM Tracking via skimage backend)
ssim_f3_noisy = ssim(f3_ground_truth, f3_input, data_range=data_range_f3)
ssim_f3_filtered = ssim(f3_ground_truth, f3_denoised, data_range=data_range_f3)
gain_f3_ssim = ssim_f3_filtered - ssim_f3_noisy

# Print unified performance summary report log on execution stream
print(f" --- QUANTITATIVE PERFORMANCE SUMMARY: REALISTIC DIGITAL TWIN CASE STUDY (F3 BLOCK) ---")
print(f"   [MSE Metric] Initial: {mse_f3_noisy:.4f}  -->  Final: {mse_f3_filtered:.4f}  (Improvement: {improvement_f3_mse:.2f}%)")
print(f"   [PSNR Metric] Initial: {psnr_f3_noisy:.2f} dB -->  Final: {psnr_f3_filtered:.2f} dB (Net Gain: +{gain_f3_psnr:.2f} dB)")
print(f"   [SSIM Metric] Initial: {ssim_f3_noisy:.4f}  -->  Final: {ssim_f3_filtered:.4f}  (Net Gain: +{gain_f3_ssim:.4f})")
print(f"----------------------------------------------------------------------------------------\n")
