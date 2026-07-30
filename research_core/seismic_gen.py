import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Script 1
def seismic_data_generation(t, x, peak_1, peak_2, peak_3, axis, height, traces, m, b, 
                            x_fault=128, throw=0.15): 
    """
    Data Generator for IEEE. 
    Implements 7 layers 
    and fully decorrelated two-dimensional noise to prevent data leakage.
    """
    # Central geological fault mask
    fault_mask = (x >= x_fault)
    t_shifted = t[:, None] - np.where(fault_mask[None, :], throw, 0.0)
    
    # Baseline geometries modeling
    valley = axis + height * np.sin(2 * np.pi * x / traces) 
    dipping = m * x + b 
    
    # Extra harmonic sinusoidal layers
    sinusoidal_extra_1 = (axis + 2.5) + (height * 1.2) * np.sin(2 * np.pi * x / traces + np.pi/4)
    sinusoidal_extra_2 = (axis + 4.5) + (height * 0.9) * np.sin(2 * np.pi * x / traces + np.pi/2)

    # 7-layer stratigraphy construction
    layer_1 = peak_1 * np.exp(-((t_shifted - 1.5) / 0.12)**2)             
    layer_2 = peak_2 * np.exp(-((t_shifted - valley[None, :]) / 0.18)**2)    
    layer_3 = peak_3 * np.exp(-((t_shifted - dipping[None, :]) / 0.12)**2) 
    layer_4 = 1.1 * np.exp(-((t_shifted - 6.8) / 0.14)**2)                 
    layer_5 = 0.8 * np.exp(-((t_shifted - 9.0) / 0.15)**2)                 
    layer_6 = 1.3 * np.exp(-((t_shifted - sinusoidal_extra_1[None, :]) / 0.16)**2) 
    layer_7 = 0.9 * np.exp(-((t_shifted - sinusoidal_extra_2[None, :]) / 0.14)**2) 
    
    # Clean signal fusion and standardization (Ground Truth)
    layers = layer_1 + layer_2 + layer_3 + layer_4 + layer_5 + layer_6 + layer_7
    layers = (layers - np.mean(layers)) / (np.std(layers) + 1e-8) 
    
    # Baseline timestamps for noisy events
    base_timestamps = [1.2, 2.4, 4.0, 5.5, 7.5]
    
    # Noise 1 (Input) - Stochastic and independent 2D matrix
    noise_1 = np.random.normal(0, 0.5, layers.shape) 
    for i in base_timestamps: 
        sign_1 = np.random.choice([-1, 1])
        n_1 = sign_1 * np.random.uniform(0.2, 0.4) 
        jitter_1 = np.random.uniform(-0.15, 0.15)
        random_space_1 = np.random.uniform(0.5, 1.5, layers.shape) 
        noise_1 += n_1 * np.exp(-((t[:, None] - (i + jitter_1)) / 0.05)**2) * random_space_1
    noisy_1 = layers + noise_1   
    
    # Noise 2 (Target) - Stochastic and independent 2D matrix
    noise_2 = np.random.normal(0, 0.5, layers.shape) 
    for i in base_timestamps: 
        sign_2 = np.random.choice([-1, 1])
        n_2 = sign_2 * np.random.uniform(0.2, 0.4) 
        jitter_2 = np.random.uniform(-0.15, 0.15)
        random_space_2 = np.random.uniform(0.5, 1.5, layers.shape) 
        noise_2 += n_2 * np.exp(-((t[:, None] - (i + jitter_2)) / 0.05)**2) * random_space_2
    noisy_2 = layers + noise_2 
    
    # Final standardizations
    noisy_1 = (noisy_1 - np.mean(noisy_1)) / (np.std(noisy_1) + 1e-8)
    noisy_2 = (noisy_2 - np.mean(noisy_2)) / (np.std(noisy_2) + 1e-8)
    
    return noisy_1.astype(np.float32), noisy_2.astype(np.float32), layers.astype(np.float32)
