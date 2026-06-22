# Deep Learning CT Reconstruction

This repository contains the artificial intelligence engine that pairs with the `DATACREATION` physics simulation. 

Our focus is entirely on the **State-of-the-Art Dual-Domain CT Reconstruction Architecture**, which is the subject of our upcoming conference paper.

## 1. Dual-Domain Pipeline (`dual-domain`)
Instead of post-processing images after the classical math has ruined them, we use an architecture that directly bridges the Sensor (Sinogram) Domain and the Image Domain.

**The process:**
1. A Deep Convolutional Network (`SinogramNet`) removes photon-starvation noise and corrects the raw TIFF projections in the Sensor Domain.
2. A `DifferentiableBackprojection` layer physically translates the 2D sinogram into a 3D grid, preserving all learned gradients.
3. A second Convolutional Network (`ImageNet`) refines the output, restoring high-frequency structures and boundaries.

## Structure

- **`scripts/run_pipeline.py`**: The master orchestration script that automatically discovers simulated datasets, builds the combined dual-domain `.npz` records, and trains the Dual-Domain network.
- **`scripts/dual_domain/`**: The dedicated dataset builder and PyTorch training loop for the dual-domain model.
- **`src/ct_recon/train_dual_domain.py`**: The core PyTorch class (`DualDomainNet`) defining the network architecture.

## How to Run

1. **Activate Environment & Install PyTorch:**
   Ensure you have a Python virtual environment activated containing `torch`, `torchvision`, and `numpy`.

2. **Train on a Single Specific Dataset (e.g., a low dose test):**
   ```bash
   python scripts/run_pipeline.py dual-domain \
       --sample FINAL30_low_dose_high_noise \
       --epochs 100
   ```

3. **Train on ALL Datasets at once:**
   (Make sure you have run the physics simulator first so that the `data/` folder is populated).
   ```bash
   python scripts/run_pipeline.py dual-domain --sample all --epochs 100
   ```

## Notes for the Conference Paper
When porting this code to the supercomputer, ensure you implement the `DifferentiableBackprojection` using an actual CUDA kernel (e.g., via the ASTRA Toolbox PyTorch wrapper or `torch-radon`). In this codebase, it is represented as a skeletal placeholder to allow for structural testing before final deployment.
