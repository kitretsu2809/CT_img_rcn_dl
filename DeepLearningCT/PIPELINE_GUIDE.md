# Dual-Domain Reconstruction Pipeline Guide

This guide provides the exact step-by-step master plan to go from an empty directory to getting State-of-the-Art Dual-Domain results for your conference paper.

This pipeline is fully automated and designed to run inside the `ct_pipeline` conda environment.

## Step 1: Generate the Physical Training Data

Before training, you must simulate the physical X-Ray projections. The ASTRA-Toolbox physics simulator will shoot "virtual X-Rays" through your 3D STL files to generate the projection data and the `settings.cto` file defining the specific scanner geometry for each sample.

**Run this in your terminal:**
```bash
conda activate ct_pipeline
cd ~/Desktop/CT_img_rcn_dl/DATACREATION
python generate_datasets.py
```
*What this does: It outputs hundreds of physical TIFF projection images and creates the specific scanner geometry (`settings.cto`) for each sample.*

---

## Step 2: Train the Dual-Domain Network

Once the data exists, trigger the master orchestrator script. This orchestrator handles everything required to train the AI.

**Run this in your terminal:**
```bash
cd ~/Desktop/CT_img_rcn_dl/DeepLearningCT
python scripts/run_pipeline.py dual-domain --sample all --epochs 50
```

**What this does under the hood:**
1. **Classical Baseline:** Runs standard FDK math on the dense projections to create perfect, high-quality 3D volumes. These act as the "ground truth" targets for the AI.
2. **Dataset Building (`01_build_dataset.py`):** Artificially throws away a majority of the projections to simulate a "fast, low-dose scan". Packages these sparse sinograms alongside the high-quality target images into a `.npz` file, and securely attaches the `settings.cto` geometry.
3. **Training (`02_train_model.py`):** The PyTorch network boots up. It reads the `settings.cto` file to dynamically configure the `torch-radon` physics layer. The AI looks at the low-dose (noisy) sinogram, tries to reconstruct it, compares it to the high-quality FDK target, and updates its weights. It saves the best model as `best_model.pt`.

---

## Step 3: Run Inference (Getting Your Results)

After the model finishes training, test it on a fresh, unseen scan to prove it generalizes for your paper. 

Assuming you have a test folder (e.g., `data/test_sample/`) containing projections and its own `settings.cto`:

**Run this in your terminal:**
```bash
python scripts/dual_domain/03_inference.py \
    --input-dir data/test_sample \
    --checkpoint outputs/dual_domain_all_datasets/best_model.pt \
    --output-file results_for_paper/deep_learning_reconstruction.tif
```
*What this does: It boots up the physics math for that specific `test_sample` geometry, pushes the low-dose projections through the trained model, and outputs a beautifully reconstructed 3D TIFF volume.*

---

## Step 4: Making Figures for your Paper

For your paper, reviewers expect visual comparisons. 
Load the Classical FDK TIFF and your new Deep Learning TIFF into a viewer like **ImageJ/Fiji** or Python's `matplotlib`. 

Place them side-by-side:
- **Image A (Classical FDK):** Show how low-dose data produces massive streak artifacts and blurring.
- **Image B (Your Dual-Domain AI):** Show how your AI mathematically bridges the sensor/image domains to eliminate streaks and restore fine details!


## TIP : generating settings.cto file from .ctdata etc files

- python DeepLearningCT/scripts/nikon_converter/convert_nikon_to_cto.py path/to/your/scan.xtekct --output-dir path/to/save/folder
