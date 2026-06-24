# Deep Learning for CT Image Reconstruction
**Project Progress & Resource Constraint Report**

## 1. Work Completed to Date

We have successfully engineered and codified the foundational architecture of the Deep Learning CT Reconstruction pipeline. Key technical milestones achieved include:

* **Shift to SOTA Dual-Domain Architecture:** 
  We initially explored purely Image-Domain methods (e.g., standard U-Nets), but identified that they struggle to map physical hardware artifacts and noise directly from the final image, and they suffer from persistent streak artifacts if dealing with highly sparse (accelerated) projections. To solve this, we architected a state-of-the-art **Dual-Domain Neural Network** that simultaneously denoises in the Sensor Domain (sinogram space) and refines in the Image Domain, enhancing full dense scans while allowing optional sparse-view acceleration.
  
* **Differentiable Physics Engine:** 
  We successfully integrated a PyTorch-native `DifferentiableBackprojection` layer using the `torch-radon` library. This allows our neural network to mathematically perform GPU-accelerated classical backprojection directly within the forward pass, anchoring the AI to real-world physics rather than forcing it to guess geometries.
  
* **Dynamic Geometry Generalization:** 
  We engineered the pipeline to be scanner-agnostic. The network dynamically loads physical scanner metadata (e.g., Source-to-Detector Distance, pixel pitch, projection counts) from `settings.cto` files at runtime. 
  
* **Nikon Geometry Parser:** 
  We developed an automated Python utility (`convert_nikon_to_cto.py`) to scrape metadata directly from proprietary Nikon `XT H 225 ST 2x` files (`.xtekct` / `.ctdata`), enabling seamless deployment on our university's specific hardware.

* **End-to-End Orchestration:** 
  A fully automated pipeline script (`run_pipeline.py`) has been written to handle dataset building (sparse downsampling), classical FDK target generation, PyTorch model training, and a final inference script (`03_inference.py`) to generate deployable 3D `.tif` volumes.

---

## 2. Training Methodology & Mathematical Reasoning

Our transition to a Dual-Domain pipeline and the design of our training process were driven by several critical observations in CT reconstruction physics:

* **Why Image-Domain Fails for Sparse Data:**
  When an industrial/aerospace CT scan is performed with a low number of projections to accelerate the scan time of dense space equipment, classical analytical methods like FDK produce severe streak artifacts due to violations of the Shannon-Nyquist sampling theorem. Standard Image-Domain neural networks struggle to map these global streaks to a clean image because the streaks are highly non-local—a single missing projection affects the entire reconstructed volume.

* **The Dual-Domain Solution:**
  Instead of trying to fix the streaks after they are created, our pipeline intervenes *before* reconstruction. 
  1. **Sensor Domain (Sinogram) Denoising:** The sparse input sinograms are passed through a U-Net to interpolate and repair the missing angular data directly in the projection space.
  2. **Differentiable Backprojection:** The repaired sinogram is mathematically projected into the Image Domain using our custom `torch-radon` layer.
  3. **Image Domain Refinement:** A secondary network cleans up any residual artifacts in the 2D slice.
  
* **Training Process Formulation:**
  The model is trained end-to-end using supervised learning. 
  - **Inputs:** Noisy, artifact-laden sinograms mimicking raw sensor data from full lab scans (or artificially downsampled sparse sinograms simulating fast scans).
  - **Targets:** High-fidelity, mathematically perfect reference reconstructions acting as the "ground truth."
  - **Loss Function:** We utilize an L1 Loss comparing both the intermediate sinogram output and the final image output against the targets. This forces the network to learn the physical mapping rather than just memorizing textures.

---

## 3. Work Remaining (Pending Hardware / Resource Execution)

While the software architecture is 100% complete and ready to run, the following execution steps are outstanding due to current local resource and compute constraints:

* **Physical Data Simulation (Bottleneck: CPU/RAM Runtime):** 
  We must run the ASTRA-Toolbox scripts to generate the simulated X-Ray projections from our STL models. Firing thousands of virtual rays through multiple high-resolution 3D models takes significant dedicated processing time and RAM before training can begin.
  
* **Deep Learning Training (Bottleneck: GPU Compute Availability):** 
  Training the Dual-Domain network is highly compute-intensive. Processing both Sinogram and Image domains simultaneously requires substantial VRAM. The network must run for ~50 epochs over the massive combined dataset, which will require dedicated, uninterrupted GPU runtime (e.g., overnight or multi-day processing on a dedicated server).
  
* **Evaluation & Benchmarking:** 
  Once the model is fully trained, we must execute the inference script on unseen test geometries and calculate quantitative metrics (PSNR, SSIM) against classical FDK baselines to validate the theorized improvements for the final publication.
