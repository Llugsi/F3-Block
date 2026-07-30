# [Script 10] Industrial linear prediction F-X Deconvolution filter (Scipy-backed)
import numpy as np
from scipy.linalg import toeplitz

def fx_deconvolution(noisy_section, filter_length=8, mu=1e-3):
    """
    Linear Prediction F-X Filter (Industrial Geophysical Standard Baseline).
    Attenuates random noise by estimating spatial reflector line coherency.
    """
    num_samples, num_traces = noisy_section.shape
    # 1. Apply Fast Fourier Transform across the synchronous time axis
    fx_data = np.fft.fft(noisy_section, axis=0)
    filtered_fx = np.zeros_like(fx_data, dtype=complex)
    
    # Process positive frequency slices due to conjugate symmetry tracking
    half_frequencies = num_samples // 2 + 1
    
    for f in range(1, half_frequencies):
        # Retrieve spatial data slice array for frequency channel 'f'
        spatial_vector = fx_data[f, :]
        filtered_spatial = np.zeros_like(spatial_vector, dtype=complex)
        
        N = num_traces
        L = filter_length
        M = N - L
        
        if M <= L:
            filtered_fx[f, :] = spatial_vector # Skip if section width is below filter length
            continue
            
        # Structure observation tracking matrices (Hankel-Toeplitz Data Matrix)
        A = np.zeros((M, L), dtype=complex)
        for i in range(M):
            A[i, :] = spatial_vector[i:i+L]
            
        y = spatial_vector[L:N] # Spatial data targets to predict forward
        
        # Solve regularized normal Gauss equations using least-squares routines
        # (A^H * A + mu * I)^(-1) * A^H * y
        AH_A = np.dot(A.conj().T, A)
        AH_y = np.dot(A.conj().T, y)
        I = np.eye(L)
        
        try:
            prediction_filter = np.linalg.solve(AH_A + mu * I, AH_y)
            
            # Apply prediction parameters to data vector
            filtered_spatial[:L] = spatial_vector[:L] # Handle initialization margins
            for i in range(L, N):
                filtered_spatial[i] = np.dot(spatial_vector[i-L:i], prediction_filter)
        except np.linalg.LinAlgError:
            filtered_spatial = spatial_vector
            
        filtered_fx[f, :] = filtered_spatial
        # Enforce structural conjugate symmetry for negative frequencies
        if f < num_samples - f:
            filtered_fx[num_samples - f, :] = filtered_spatial.conj()

    # 2. Return data vectors into space-time domain matrices (t-x)
    denoised_section = np.fft.ifft(filtered_fx, axis=0).real
    return denoised_section.astype(np.float32)
