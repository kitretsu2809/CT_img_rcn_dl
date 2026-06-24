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

1. **Activate Conda Environment:**
   Ensure you have your unified `ct_pipeline` Conda environment activated. (See the main `PROJECT_MANUAL.md` for setup instructions).
   ```bash
   conda activate ct_pipeline
   ```

2. **Train the AI on ALL Generated Datasets (Standard Workflow):**
   *(Make sure you have run the physics simulator first so that the `data/` folder is populated with scans).*
   
   To train a robust AI, we feed it every single dataset we generated (standard scan, fast scan, different CAD models, etc.). You only need to run this one command:
   ```bash
   python scripts/run_pipeline.py dual-domain --sample all --epochs 100
   ```
   **What this does automatically:**
   - It scans the `data/` directory and finds all your generated scans.
   - It builds a massive unified dataset from all those scans.
   - It trains the Dual-Domain network for 100 iterations (epochs).
   - It saves the final trained AI model to `outputs/dual_domain_all_datasets/`.

## Notes for the Conference Paper
The architecture in `train_dual_domain.py` is fully implemented and **automatically** integrates the differentiable CUDA kernel (`torch-radon`) if it is installed in your Conda environment. Make sure to follow the *Phase 0 Supercomputer Setup* instructions in `PROJECT_MANUAL.md` so PyTorch can pass mathematical gradients seamlessly between the Sensor Domain and the Image Domain. Once the library is installed, the code handles the rest to give you State-of-the-Art results for your paper!
