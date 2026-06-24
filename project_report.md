# Deep Learning for High-Resolution Industrial CT Artifact Reduction
**Formal Project Report (Draft)**

---

## 1. Abstract
Computed Tomography (CT) is an essential non-destructive imaging technique used extensively in aerospace and industrial inspection (e.g., space equipment). In these lab environments, full-density scans (using all available projections) are the standard to ensure maximum micro-resolution. However, industrial CT of dense parts often suffers from severe beam hardening, metal artifacts, and sensor noise that traditional analytical reconstruction algorithms, such as the Feldkamp-Davis-Kress (FDK) algorithm, struggle to fully resolve. Furthermore, the ability to optionally accelerate these scans via sparse-view acquisition is highly desirable for high-throughput scenarios. This project outlines the development and implementation of a state-of-the-art **Dual-Domain Deep Learning Architecture** designed to reconstruct and enhance high-fidelity 3D volumes from both full dense projection data (for artifact reduction) and sparse projection data by leveraging both deep neural networks and deterministic physical backprojection.

## 2. Work Completed: System Architecture & Data Pipeline

### 2.1 The Problem with Image-Domain Learning
Early experiments in this project involved training purely Image-Domain Convolutional Neural Networks (like standard U-Nets) to map low-quality FDK reconstructions to high-quality targets. However, we identified a critical limitation: because the streak artifacts caused by missing angles are highly non-local (global across the image), standard CNNs struggle to effectively learn the mapping.

### 2.2 The Limitations of Direct Sinogram-to-Image Learning
An initial proposition in this project was to use Deep Learning to reconstruct the 3D volume *solely* from the raw sinogram, entirely bypassing analytical physics. While conceptually appealing (often referred to in literature as Direct Transform Learning), this approach suffers from two fatal flaws:
1. **The Curse of Dimensionality:** A sinogram mapping directly to an image requires dense, fully-connected layers because every point in a projection contributes globally to the entire image space. For industrial, high-resolution CT scans, a single dense layer bridging the two domains would require billions of parameters, instantly exhausting GPU memory limits.
2. **Loss of Physical Priors:** By discarding the deterministic mathematics of the inverse Radon transform, the neural network is forced to blindly "re-learn" the laws of physics and projection geometry from scratch. This is computationally inefficient and results in blurry, hallucinated textures that fail to generalize to unseen hardware configurations.

### 2.3 The Dual-Domain Solution
To overcome the limitations of pure Image-Domain learning in removing complex physical artifacts (such as those from dense metal or optional sparse scanning), we engineered an end-to-end Dual-Domain network that intervenes in both the Sensor Domain and the Image Domain, bridged by classical physics.

1. **Sensor Domain (Sinogram) Denoising:** 
   The input sinograms (whether full dense scans with physical noise or accelerated sparse scans) are first passed through a Deep Learning U-Net (`sino_net`). This network is tasked with cleaning up physical noise, repairing scattered data, or interpolating missing angular data directly in the projection space, before reconstruction errors can propagate globally.
   
2. **Differentiable Physics Bridge:** 
   Crucially, the transformation from the Sinogram to the Slice Image is *not* "guessed" by an AI. Instead, the repaired sinogram is passed to a custom `DifferentiableBackprojection` layer powered by the `torch-radon` library. This layer executes deterministic GPU-accelerated classical math (FBP/FDK principles) to perform the physical backprojection. By making this mathematical step fully differentiable, gradients can flow backward through the physics engine during training, allowing the AI to learn physical reality.

3. **Image Domain Refinement:** 
   Because the backprojection step is mathematically perfect but may carry over some residual noise from the sparse data, a secondary Deep Learning network (`image_net`) takes that raw mathematical reconstruction and performs a final clean-up to produce the refined output slice.

### 2.4 Dynamic Geometry Generalization & Automation
A major hurdle in deep learning for CT is rigid geometry. To make our pipeline scanner-agnostic, we implemented a dynamic geometry loading system.
* **Metadata Parsing:** An automated utility (`convert_nikon_to_cto.py`) was written to parse proprietary metadata from Nikon `XT H 225 ST 2x` scanners (`.xtekct` / `.ctdata` files) into a standardized `settings.cto` format.
* **Dynamic Network Initialization:** The Dual-Domain network automatically reads the `settings.cto` file at runtime, dynamically configuring the Source-to-Detector Distances, pixel pitches, and projection counts for the `torch-radon` physics engine. This ensures the model mathematically adapts to any unseen scanner hardware.

An overarching orchestrator script (`run_pipeline.py`) was also built to seamlessly manage classical target generation, sparse dataset downsampling, network training, and a final inference pipeline (`03_inference.py`).

---

## 3. Results and Evaluation

*(Space reserved for quantitative and qualitative analysis post-training)*

### 3.1 Quantitative Metrics
*Placeholder for Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index (SSIM) comparisons between Classical FDK and the Dual-Domain Deep Learning reconstructions.*

### 3.2 Visual Comparisons
*Placeholder for side-by-side slice image figures demonstrating the elimination of streak artifacts.*

---

## 4. Conclusion and Future Work

*(Space reserved for final summary of findings and deployment recommendations)*
