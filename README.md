# Self-Supervised Suppression of Spatially Coherent Seismic Noise via
# Directional Anisotropic Causal Networks: Application to the F3
# Block

Official repository containing the source code, network architectures, validation pipelines, and scientific visualization scripts for the methodology presented in our manuscript.

[![Python 3.8+](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)
[![License: MIT](https://shields.io)](https://opensource.org)

---

## 📊 Overview & Core Contributions

This research introduces a self-supervised seismic data denoising paradigm based on strict **Anisotropic Blind-Spot Constraints** under the Noise2Noise framework. By physically depriving the network of the identity pixel information during training, our framework avoids cross-branch data leakage and ensures convergence toward structural geological reflectors without requiring real ground-truth data.

### Key Architectural Highlights:
1. **Hermetic Causal Streams:** Four independent directional streams (North, South, East, West) implemented via strict asymmetric truncation padding to isolate information propagation.
2. **Dynamic Attention Gating:** Independent 1x1 convolutional operators that weight cardinal features without breaking spatial blind-spot guarantees.
3. **Field Generalization:** Evaluated successfully on synthetic stratigraphic models (7 harmonic layers with faults) and real production field data from the **North Sea F3 Block (Doggerland)**.

---

## 📁 Repository Structure & Script Mapping

The repository is modularly divided into two functional groups to streamline code execution and peer review:

```bash
├── 📜 README.md                             # Repository documentation and review guide
├── 📂 research_core/                        # Group 1: Research Generation & Pipelines
│   ├── seismic_gen.py                       # [Script 1] 7-layer synthetic stratigraphic and noise simulator
│   ├── dataset.py                           # [Script 2 & 4] Coordinated DoggerlandN2NDataset loader (with GT/Noise support)
│   ├── blind_spot_net.py                    # [Script 3] Pure Causal Conv2D operators and BlindSpotNet architecture
│   ├── train_pipeline.py                    # [Script 4] Dual-axis training loop (Self-Supervised MSE vs Joint Real PSNR)
│   ├── infer_synthetic.py                   # [Script 5] Global synthetic inference pipeline and 3-panel metrics evaluator
│   ├── f3_digital_twin.py                   # [Script 6] F3 Block digital twin generator (Shear faults & mud noise)
│   ├── infer_f3_twin.py                     # [Script 7 & 8] Quantitative evaluation on F3 digital twin
│   ├── baseline_unet.py                     # [Script 9 & 14B] Standard UNet baseline architecture with bilinear interpolation
│   ├── classical_fx.py                      # [Script 10] Industrial linear prediction F-X Deconvolution filter (Scipy-backed)
│   ├── benchmark_table.py                   # [Script 11] Dynamic Pandas formatting module for structural IEEE report tables
│   ├── benchmark_trends.py                  # [Script 12] Dual-axis PSNR/SSIM high-density bar-chart generator
│   └── infer_field_segy.py                  # [Script 13, 14 & 14B] Industrial SEG-Y reader (Segyio) and 1x4 cross-validation
└── 📂 figures_generation/                   # Group 2: Journal Scientific Visualization Layouts
    ├── fig_causal_diagram.py                # [Script 15] 5x5 Grid vector mapping for Blind-Spot visualization
    ├── fig_attention_gate.py                # [Script 16] Hardware bus-corrected attention gate schematic
    ├── fig_patch_extraction.py              # [Script 17] Multi-index PyTorch DataLoader online stochastic sampler diagram
    ├── fig_local_f3_map.py                  # [Script 18] Single-column (3.5") localized acquisition orientation layout
    └── fig_regional_macro_map.py            # [Script 19] Macro geographic context layout with supersampled poly coastlines
```

---

## 🛠️ Environment Setup & Dependencies

The code is designed to run efficiently on both standard CPUs and CUDA-compliant GPUs. Ensure you have the following prerequisites installed:

```bash
# Clone the repository
git clone https://github.com
cd your-repo-name

# Install required packages
pip install numpy torch matplotlib pandas scipy scikit-image segyio
```

*Note: `segyio` is strictly mandatory for running the industrial field data pipelines (`infer_field_segy.py`).*

---

## Execution Guide for Reviewers

To replicate the quantitative tables and high-resolution figures presented in the manuscript, follow these execution steps sequentially:

### Step 1: Execute Synthetic Training & Model Convergence
Run the main training loop to observe the self-supervised optimization process. This script will train the network over 200 epochs and output the synchronized dual-axis convergence graph (MSE Loss vs Real Validation PSNR).
```bash
python research_core/train_pipeline.py
```

### Step 2: Validate Against Industrial Baselines (F3 Digital Twin)
Compute metrics (MSE, PSNR, SSIM) dynamically against classical F-X Deconvolution and standard UNet architectures to generate the official LaTeX-ready performance summary table.
```bash
python research_core/benchmark_table.py
```

### Step 3: Run Industrial Field Data Inferences (Field SEG-Y File)
Ensure your target production file (`Seismic_data.sgy`) is placed in the root directory. Execute the horizontal validation pipeline to parse the SEG-Y volume, perform strict Z-score normalizations, and deploy parallel network inferences.
```bash
python research_core/infer_field_segy.py
```
*Expected Outputs:* 
* `benchmark_metrics.jpg` (High-DPI bar/line trend diagram)
* `ieee_field_validation_4panel.png` (Horizontal cross-validation layout with high-visibility axes font size `18`)

### Step 4: Recompile Manuscript Schematics
To re-render the exact high-fidelity vector diagrams included in the methodology sections of the paper, execute the scripts within the visualization folder:
```bash
python figures_generation/fig_causal_diagram.py
python figures_generation/fig_attention_gate.py
python figures_generation/fig_patch_extraction.py
python figures_generation/fig_regional_macro_map.py
```

---

## 🎯 Reproducibility & Technical Parameters Summary
* **Grid Patch Size:** 64 × 64 pixels extracted stochastically via an online uniform distribution $\mathcal{U}(0, \cdot)$.
* **Training Volume:** 1,200 training patches / 300 validation patches per epoch (independent geological setups to avoid overfitting).
* **Optimization Framework:** Adam Optimizer (Learning Rate = 0.001), backed by `torch.backends.cudnn.benchmark = True`.
* **Real Seismic Dimensions:** Natively reads and processes blocks of 462 samples × 256 traces directly via direct GPU memory tensor allocation.

---

## 📜 Citation & Contacts

If you find this research or code useful for your work, please cite our official paper:

```bibtex
@article{llugsi2026anisotropic,
  title={Self-Supervised Suppression of Spatially Coherent Seismic Noise via
Directional Anisotropic Causal Networks: Application to the F3
Block},
  author={Llugsi, Ricardo},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  volume={XX},
  number={X},
  pages={XX--XX},
  doi={XX.XXXX/TGRS.2026.XXXXXXX}
}
```

For inquiries regarding corporate data integrations, hardware stream adjustments, or replication issues, please open an issue in this repository or contact the corresponding author at `ricardo.llugsi@epn.edu.ec`.

