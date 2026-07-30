# [Script 13, 14 & 14B] Industrial SEG-Y reader (Segyio) and 1x4 cross-validation
import segyio
def inferencia_seccion_real_f3_pura(ruta_segy, num_samples=462, num_traces=256, skip_traces=1000):
    """
    IEEE TGRS Inherent Field Data Inference Engine (Zero Numerical Distortions).
    Parses native 462 samples directly from industrial storage file handles
    and deploys optimized GPU-parallel convolutional routines.
    """
    print(f"⏳ Extracting real production data via sequential volume streaming: {ruta_segy}")
    trace_buffer = []
    
    with segyio.open(ruta_segy, "r", ignore_geometry=True) as segy_file:
        for i in range(skip_traces, skip_traces + num_traces):
            if i >= len(segy_file.trace):
                break
            full_trace = segy_file.trace[i]
            trace_buffer.append(full_trace[:num_samples])
            
    # Format native field matrix arrays: (Row = Time samples, Col = Spatial traces)
    real_section = np.array(trace_buffer).T 
    
    # [MANDATORY PRE-PROCESSING]: Enforce strict Z-score normalization scaling for PyTorch
    f3_real_input = (real_section - np.mean(real_section)) / (np.std(real_section) + 1e-8)
    
    # Reshape input vectors: (Batch=1, Channel=1, Height=462, Width=256)
    tensor_input = torch.tensor(f3_real_input.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        tensor_output = model(tensor_input)
        
    # Isolate inference results matrix from GPU tensors using index slicing (0,0)
    f3_real_denoised = tensor_output[0, 0].cpu().numpy()
    
    # Structural visual validation plots setup (Standard Journal Layout)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    cmap_style = 'seismic'
    
    axes[0].imshow(f3_real_input, cmap=cmap_style, aspect='auto', vmin=-2, vmax=2)
    axes[0].set_title("A) Raw Production Field Input Data\n(North Sea F3 Block)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Travel Time Samples (Z=462)", fontsize=10)
    axes[0].set_xlabel("Seismic Trace Index (x=256)", fontsize=10)
    
    axes[1].imshow(f3_real_denoised, cmap=cmap_style, aspect='auto', vmin=-2, vmax=2)
    axes[1].set_title("B) Proposed Network Output Signal\n(Reconstructed Hermetic BSN)", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Seismic Trace Index (x=256)", fontsize=10)
    
    for ax in axes:
        ax.grid(False)
        
    plt.tight_layout()
    plt.show()
    
    return f3_real_input, f3_real_denoised

# Script 14, The .sgy files used in this study were obtained from the North Sea F3 
# Block dataset via the TerraNubis portal (terranubis.com). The authors thank dGB 
# Earth Sciences for making this dataset available as an OpendTect project.
f3_in, f3_out = inferencia_seccion_real_f3_pura("Seismic_data.sgy") 
