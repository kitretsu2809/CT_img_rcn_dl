# DeepLearningCT & Synthetic Dataset Generation Pipeline

This repository contains a comprehensive "digital twin" CT projection pipeline paired with a Deep Learning (U-Net) reconstruction and enhancement framework. It is capable of simulating realistic physical X-ray geometries to generate high-fidelity synthetic training data, which is then seamlessly consumed by our deep learning models.

## 🚀 Overview

The codebase is divided into two primary domains:
1. **`DATACREATION/`**: Simulates the physical realities of a CT scanner (Cone Beam geometry, partial volume effects via downsampling, detector noise, etc.) using high-resolution STL phantoms. It exports 16-bit TIFF projections and standard `settings.cto` metadata.
2. **`DeepLearningCT/`**: A deep learning and classical reconstruction framework that auto-discovers generated datasets, reconstructs them using analytical methods (FDK/FBP), and trains deep neural networks (U-Net) to enhance or directly reconstruct the imagery.

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.8+
- [ASTRA Toolbox](https://www.astra-toolbox.com/) (Required for GPU-accelerated forward projection and FDK reconstruction)
- GPU support strongly recommended (CUDA)

### Installation
1. Clone this repository:
   ```bash
   git clone git@github.com:kitretsu2809/CT_img_rcn_dl.git
   cd CT_img_rcn_dl
   ```

2. Install the required dependencies for the Deep Learning pipeline:
   ```bash
   pip install -r DeepLearningCT/requirements.txt
   ```
   *(Ensure you have `astra-toolbox` installed, typically via conda: `conda install -c astra-toolbox astra-toolbox`)*

---

## 1️⃣ Dataset Generation (`DATACREATION`)

The generation pipeline takes `.stl` CAD models and turns them into thousands of projection images mimicking a real X-ray scanner.

### The Automation Engine (Recommended)
We provide an automated batch-generation script that iterates through different physical variations (e.g., standard dose, low dose with high noise, perfect baseline, and different downsampling factors). 

1. Drop your `.stl` files into `DATACREATION/STL/`
2. Run the generator:
   ```bash
   cd DATACREATION
   python generate_datasets.py
   ```
3. The script will automatically spin up variations and drop them into the master `data/` directory at the repository root.

### Manual Implementation (Single Variation)
If you want to manually run a single specific projection simulation, you can use the core projection script with command-line arguments:

```bash
python usaf_projection_script.py \
    --stl STL/FINAL30.stl \
    --output_dir ../data/MyCustomDataset \
    --zip_name MyCustomDataset.zip \
    --supersample 0.025 \
    --downsample 2 \
    --i0 50000.0 \
    --gaussian-std 10.0
```
- `--no-noise`: Add this flag to disable the physics noise simulation for a "perfect" reference scan.
- `--supersample` & `--downsample`: Controls the mathematical anti-aliasing to simulate realistic voxel edge boundaries.

---

## 2️⃣ Deep Learning Pipeline (`DeepLearningCT`)

Once datasets are generated into the `data/` directory, the Deep Learning pipeline automatically discovers them. You do not need to configure any paths manually.

### Running the Full Pipeline
The central orchestrator is `scripts/run_pipeline.py`. It can manage dataset building, FDK classical reconstruction, and U-Net training.

**Example: Train a U-Net on a newly generated dataset**
```bash
cd DeepLearningCT
python scripts/run_pipeline.py unet \
    --sample FINAL30_low_dose_high_noise \
    --epochs 100 \
    --batch-size 8
```
*Note: The `--sample` argument must perfectly match the name of the folder inside the `data/` directory.*

### Available Pipeline Modes
- `fdk`: Runs a classical Feldkamp-Davis-Kress (FDK) analytical reconstruction on the projections.
- `sinogram`: Trains a model focusing on sinogram-domain enhancement.
- `enhance`: Trains an image-domain enhancement model.
- `unet`: End-to-end training of a U-Net model from projections to final volumes.

---

## 📂 Directory Structure

```text
├── data/                            # (Git Ignored) Auto-discovered input datasets containing projections and settings.cto
├── outputs/                         # (Git Ignored) Trained models, logs, and reconstructed volumes
├── DATACREATION/                    
│   ├── generate_datasets.py         # The dataset batch automation engine
│   ├── usaf_projection_script.py    # Core physics simulation and ASTRA projection script
│   └── STL/                         # Place your raw CAD (.stl) files here
└── DeepLearningCT/                  
    ├── scripts/                     # Execution entry points (run_pipeline.py)
    ├── src/                         # Core libraries (ct_recon, geometry, data_loader)
    ├── requirements.txt             
    └── references/                  # Reference papers and baseline network code
```

## ⚠️ Notes on Geometry Integration
The simulated projection geometry in `DATACREATION` perfectly mirrors the physical hardware assumed in the `DeepLearningCT` reconstruction algorithms.
- **Source-to-Object Distance (SOD):** 160 mm
- **Source-to-Detector Distance (SDD):** 200 mm
- **Detector Pixel Size:** 48 µm
- **Geometry Type:** Cone Beam 3D
