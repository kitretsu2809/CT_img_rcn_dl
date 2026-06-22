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
   - As X-rays pass through the voxels, they are absorbed. We use the ASTRA Toolbox (a high-performance GPU library) to calculate the exact line-integrals of attenuation from 360 different angles. This raw sensor data is called a **Sinogram**.
4. **Detector Noise (Poisson & Gaussian):**
   - Perfect math isn't useful for AI training because real scanners have imperfect sensors and photon-starvation. We inject **Poisson noise** (to simulate the random quantum nature of X-ray photons hitting the detector) and **Gaussian noise** (to simulate electrical sensor noise).

---

## 2. The AI & ML Architecture (`DeepLearningCT`)

Once we have the raw sensor data (sinograms), we need to turn it into 3D images. 

### Classical Reconstruction (The Baseline)
Historically, CT scanners use analytical math to convert sinograms into images, primarily an algorithm called **FBP** (Filtered Back Projection) or its 3D Cone-Beam equivalent, **FDK** (Feldkamp-Davis-Kress). 
- **The Problem:** When FDK encounters the noise we injected in Step 1, the mathematical inversion process *smears* the noise across the entire image, creating harsh streak artifacts and destroying fine structural details.

### Deep Learning Enhancement (U-Net)
To fix FDK's flaws, we train an Artificial Neural Network.
- We use a **Convolutional Neural Network (CNN)** specifically structured as a **U-Net**. 
- **How it learns:** We give the network pairs of images: 
  1. The noisy, artifact-ridden FDK image (Input)
  2. A mathematically perfect, noise-free image (Target Ground Truth)
- The AI plays a game of spot-the-difference, updating its internal weights (via Backpropagation) over thousands of iterations to learn exactly how to subtract streaks and repair noise without blurring the actual object.

### State-of-the-Art Future: Dual-Domain Networks
We have also laid the foundation for **Dual-Domain** learning. A 2D U-Net is limited because FDK permanently destroys high-frequency data. A Dual-Domain network cleans the noise in the *Sensor Domain* (raw sinograms) first, uses a differentiable projection layer, and then refines the final image. This provides mathematical fidelity and represents the cutting edge of AI radiology in 2024.

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
│   ├── scripts/run_pipeline.py      # Master orchestrator for Classical & AI workflows
│   ├── src/ct_recon/                # Deep Learning architecture (U-Net, Dual-Domain)
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
*What happens:* The script will find all 3D objects (e.g., `FINAL3.stl` and `FINAL30.stl`). For each object, it will generate 4 unique physical simulations:
- `standard` (Normal noise, 50-micron resolution)
- `low_dose_high_noise` (Simulates photon-starvation with intense Gaussian noise)
- `perfect_no_noise` (Mathematically perfect reference scan)
- `downsample_4x` (100-micron resolution scan)

These will all be saved as separate, fully packaged folders inside `/home/kitretsu/Desktop/SUMMER/data/`.

### Phase 2: Train the Deep Learning Model (AI Execution)
*Once the `data/` folder is packed with variations, you can instruct the orchestrator to train the U-Net on the **entire massive collection** at once.*

```bash
# 1. Navigate to the AI directory
cd /home/kitretsu/Desktop/SUMMER/DeepLearningCT

# 2. Start the U-Net training pipeline on ALL generated datasets
python scripts/run_pipeline.py unet --sample all --epochs 50 --batch-size 8
```
*What happens:* 
1. The script auto-discovers every single physical variation generated in `../data/`.
2. It runs classical FDK reconstruction on all the noisy data to create inputs.
3. It builds training pairs (Noisy Input vs Perfect Target) for every single dataset.
4. It aggregates all these slices into one massive, randomized training pool.
5. It initializes the PyTorch U-Net and trains for 50 epochs on your GPU, saving the ultra-robust AI model to `outputs/`.

### Phase 3: Evaluate Alternative Modes
If you want to just run classical physics math without AI:
```bash
python scripts/run_pipeline.py classical --sample FINAL30_standard
```
If you want to initialize the bleeding-edge Dual-Domain architecture framework:
```bash
python scripts/run_pipeline.py dual-domain --sample FINAL30_standard
```
