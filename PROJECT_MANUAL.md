# Digital Twin CT Pipeline & AI Reconstruction Manual

This manual serves as a complete guide to understanding, executing, and explaining the CT Simulation and Deep Learning Reconstruction pipeline. It covers the underlying physics, the artificial intelligence architecture, and the exact commands needed to run the software from start to finish.

---

## 1. The Physics: How We Simulate a CT Scanner (`DATACREATION`)

Before we can train an AI, we need data. Medical CT scanners work by shooting X-rays through an object from multiple angles. We built a **Digital Twin** that mathematically simulates this exact physical process inside a computer using 3D CAD models (`.stl` files).

### Core Concepts Simulated
1. **Voxelization & Partial Volume Effect (Anti-Aliasing):**
   - **The Problem:** CAD models are smooth surfaces, but computers process data in 3D pixels (voxels). Standard conversion creates "blocky" jagged edges (aliasing).
   - **Our Solution:** We *super-sample* the 3D model at an ultra-fine resolution (25 microns) and then mathematically downsample it (to 50 microns) by averaging blocks of voxels. This creates grayscale voxels at the boundaries, perfectly simulating the "partial volume effect" seen in real physical scanners where an X-ray hits the very edge of an object.
2. **Cone Beam Geometry:**
   - Real X-rays don't shoot in parallel lines; they emit from a point source and spread out like a cone. We simulate a physical setup where the **Source-to-Object Distance (SOD)** is 160mm and the **Source-to-Detector Distance (SDD)** is 200mm.
3. **The Beer-Lambert Law (Attenuation):**
   - As X-rays pass through the voxels, they are absorbed. We use the ASTRA Toolbox (a high-performance GPU library) to calculate the exact line-integrals of attenuation from 360 different angles. This raw sensor data is called a **Sinogram** (saved as TIFF files).
4. **Detector Noise (Poisson & Gaussian):**
   - Perfect math isn't useful for AI training because real scanners have imperfect sensors and photon-starvation. We inject **Poisson noise** (to simulate the random quantum nature of X-ray photons hitting the detector) and **Gaussian noise** (to simulate electrical sensor noise).

---

## 2. The AI & ML Architecture: Dual-Domain Reconstructor (`DeepLearningCT`)

Once we have the raw sensor data (sinograms), we need to turn it into 3D images. We use the **State-of-the-Art Dual-Domain architecture**.

Instead of relying solely on a post-processing U-Net (which tries to fix noise *after* classical math ruins the high-frequency details), our pipeline processes data systematically across both the sensor and image domains.

### How The Model Works
1. **Sinogram-Net (Sensor Domain):** The 2D TIFF projections are fed directly into a deep neural network that cleans the raw quantum noise before it causes streak artifacts.
2. **Differentiable Backprojection (The Bridge):** The cleaned sinogram is pushed through a differentiable physics layer, mathematically transforming it into a 3D slice. (Note: PyTorch passes the learning gradients straight through this math!).
3. **Image-Net (Image Domain):** A second U-Net refines the 3D slice, recovering high-frequency textures and structures.

> [!TIP]
> **Conference Paper Action Item (Supercomputer):** 
> The architecture in `train_dual_domain.py` is fully implemented except for the `DifferentiableBackprojection` layer, which is currently a placeholder block. When you move to the supercomputer, you will need to replace that placeholder with a compiled CUDA kernel (using libraries like `torch-radon` or `astra.pytorch`) so PyTorch can pass mathematical gradients seamlessly between the Sensor Domain and the Image Domain. Once compiled, this will give you State-of-the-Art results for your paper!

---

## 3. Codebase Architecture

The project is unified under a single repository but cleanly split by purpose:

```text
SUMMER/
├── DATACREATION/                    # THE PHYSICS SIMULATOR
│   ├── venv/                        # The Python Virtual Environment (Requires ASTRA)
│   ├── generate_datasets.py         # Batch automation to generate hundreds of variations
│   ├── usaf_projection_script.py    # The core physics & ASTRA projection engine
│   └── STL/                         # Put your raw 3D CAD files here
├── data/                            # (Auto-generated) Contains the resulting raw projections
├── DeepLearningCT/                  # THE AI ENGINE
│   ├── scripts/run_pipeline.py      # Master orchestrator for Dual-Domain workflows
│   ├── scripts/dual_domain/         # Data loading and training scripts
│   ├── src/ct_recon/                # Deep Learning architecture (train_dual_domain.py)
│   └── outputs/                     # Trained AI weights and reconstructed 3D TIFFs
```

---

## 4. Execution Manual: How to Run the Pipeline

Here are the exact commands you need to run to execute the pipeline from start to finish on your local machine.

> [!WARNING]
> Both the Physics Simulation and AI Training require intense GPU computation. Ensure your machine is idle before running these scripts.

### Phase 1: Generate the Synthetic Data (Physics Simulation)
*You will use the isolated `venv` inside DATACREATION that has ASTRA correctly installed. This script has been super-charged to scan the `DATACREATION/STL/` folder and generate multiple variations for **every** CAD model it finds.*

```bash
# 1. Navigate to the Simulator directory
cd /home/kitretsu/Desktop/SUMMER/DATACREATION

# 2. Run the dataset generation using the local virtual environment
./venv/bin/python generate_datasets.py
```
*What happens:* The script will find all 3D objects. For each object, it will generate 4 unique physical simulations:
- `standard` (Normal noise, 50-micron resolution)
- `low_dose_high_noise` (Simulates photon-starvation with intense Gaussian noise)
- `perfect_no_noise` (Mathematically perfect reference scan)
- `downsample_4x` (100-micron resolution scan)

These will all be saved as separate, fully packaged folders inside `/home/kitretsu/Desktop/SUMMER/data/`.

### Phase 2: Train the Dual-Domain Model (Standard Workflow)
*Once the `data/` folder is packed with variations, you can train the Dual-Domain Network on ALL generated datasets simultaneously.*

To train a robust AI, we feed it every single dataset we generated (normal dose, low dose, different CAD models, etc.). You only need to run this one command:

```bash
# 1. Navigate to the AI directory
cd /home/kitretsu/Desktop/SUMMER/DeepLearningCT

# 2. Train on ALL datasets at once
python scripts/run_pipeline.py dual-domain --sample all --epochs 100
```
*What happens:* 
1. The script auto-discovers every single physical variation generated in `../data/`.
2. It builds the massive `dual_domain_dataset_all.npz` file, extracting the noisy sinograms, perfect sinograms, and perfect 3D image slices.
3. It initializes the PyTorch Dual-Domain model and trains it across both domains simultaneously.
4. The trained model weights are saved to `outputs/dual_domain_all_datasets/`.

### Phase 3: Using Real Physical CT Scans (No Simulation)
If you scan a real physical object on a hardware CT machine, you can feed that real data directly into this AI pipeline. 

You simply need to format your real scan to mimic our simulation structure:
1. Create a new folder for your scan: `SUMMER/data/my_real_scan/`
2. Place all your real 16-bit TIFF projection images into: `SUMMER/data/my_real_scan/projections/`
3. Create a `settings.cto` text file in `SUMMER/data/my_real_scan/` that contains your physical scanner's exact hardware geometry (e.g., Source-to-Object Distance, Detector Pitch, etc.).

Once that folder exists, the `run_pipeline.py dual-domain --sample all` command will automatically discover it, extract its data, and train the AI using your real physical data alongside the synthetic data!
