# Graph Report - SUMMER  (2026-06-23)

## Corpus Check
- 50 files · ~24,416 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 374 nodes · 717 edges · 29 communities (21 shown, 8 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `371ff725`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 28|Community 28]]

## God Nodes (most connected - your core abstractions)
1. `resolve_repo_path()` - 27 edges
2. `DegradedProjectionData` - 17 edges
3. `parse_geometry()` - 16 edges
4. `reconstruct_volume_from_projection_dataset()` - 16 edges
5. `make_full_projection_dataset()` - 15 edges
6. `load_sample()` - 14 edges
7. `CTScanData` - 13 edges
8. `ndarray` - 11 edges
9. `CT_uitil` - 10 edges
10. `main()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `DegradedProjectionData` --uses--> `DegradedProjectionData`  [INFERRED]
  DeepLearningCT/src/ct_recon/reconstruct_fdk_astra.py → DeepLearningCT/src/ct_recon/simulate_degradation.py
- `run_sample2_reconstruction()` --calls--> `set_sample()`  [EXTRACTED]
  DeepLearningCT/scripts/classical_reconstruction/reconstruct_fdk_sample2.py → DeepLearningCT/src/ct_recon/paths.py
- `get_sample_config()` --calls--> `load_cto_settings()`  [INFERRED]
  DeepLearningCT/scripts/common/sample_config.py → DeepLearningCT/src/ct_recon/data_loader.py
- `DegradedProjectionData` --uses--> `DegradedProjectionData`  [INFERRED]
  DeepLearningCT/src/ct_recon/build_training_pairs.py → DeepLearningCT/src/ct_recon/simulate_degradation.py
- `DegradedProjectionData` --uses--> `CTScanData`  [INFERRED]
  DeepLearningCT/src/ct_recon/simulate_degradation.py → DeepLearningCT/src/ct_recon/data_loader.py

## Import Cycles
- None detected.

## Communities (29 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.07
Nodes (40): copy_pair_folder(), export_training_pairs(), main(), resolve_repo_path(), set_sample(), DoubleConv, _import_torch_or_exit(), load_sparse_dataset() (+32 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (35): ConfigParser, CasePreservingConfigParser, CTScanData, list_projection_files(), load_cto_settings(), load_projection_stack(), load_sample(), load_sample1() (+27 more)

### Community 2 - "Community 2"
Cohesion: 0.20
Nodes (25): align_pair_shapes(), build_default_training_pairs(), build_pair_for_dataset(), center_crop_to_match(), normalize_volume(), save_pair_outputs(), add_gaussian_noise(), add_poisson_noise() (+17 more)

### Community 3 - "Community 3"
Cohesion: 0.20
Nodes (18): Run FDK reconstruction for sample_2., run_sample2_reconstruction(), block_average_2d(), build_cone_geometry(), build_volume_geometry(), convert_to_attenuation(), crop_valid_z(), downsample_projection_stack() (+10 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (10): DifferentiableBackprojection, DoubleConv, DualDomainNet, ImageNet, main(), Subnetwork operating in the Sensor (Sinogram) Domain.     Cleans up noise and re, Placeholder for a differentiable Radon transform (FBP).     In SOTA implementati, Subnetwork operating in the Image Domain.     Refines the FBP-reconstructed imag (+2 more)

### Community 5 - "Community 5"
Cohesion: 0.22
Nodes (17): get_sample_config(), get_sample_paths(), Universal sample configuration., Get configuration for a sample., Get all paths for a sample., Path, main(), Run direct sinogram reconstruction pipeline. (+9 more)

### Community 6 - "Community 6"
Cohesion: 0.22
Nodes (8): decode(), make_model_3(), psnr(), sinLayer, train(), train_step(), u_function(), w_bfunction()

### Community 7 - "Community 7"
Cohesion: 0.23
Nodes (9): decode(), interp(), make_model_3(), psnr(), sinLayer, train(), train_step(), u_function() (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (13): 1️⃣ Dataset Generation (`DATACREATION`), 2️⃣ Deep Learning Pipeline (`DeepLearningCT`), Available Pipeline Modes, DeepLearningCT & Synthetic Dataset Generation Pipeline, 📂 Directory Structure, Installation, Manual Implementation (Single Variation), ⚠️ Notes on Geometry Integration (+5 more)

### Community 9 - "Community 9"
Cohesion: 0.26
Nodes (8): BN(), build_unpool(), DenseNet(), make_model(), max_pool(), mix(), train(), train_step()

### Community 10 - "Community 10"
Cohesion: 0.32
Nodes (11): append_face(), carve_internal_resolution_block(), export_resolution_cuboid_stls(), generate_resolution_cuboid_voxels(), mm_to_voxels(), Carve fully enclosed alternating void lamellae inside a chamber.     The chamber, triangle_normal(), voxel_surface_triangles() (+3 more)

### Community 11 - "Community 11"
Cohesion: 0.29
Nodes (10): build_transforms(), _import_torch_or_exit(), load_slice_records(), main(), psnr(), save_history(), SliceRecord, split_records() (+2 more)

### Community 12 - "Community 12"
Cohesion: 0.24
Nodes (6): decode(), make_model_3(), psnr(), sinLayer, train(), train_step()

### Community 14 - "Community 14"
Cohesion: 0.20
Nodes (9): 1. Classical (`classical`), 2. Sinogram (`sinogram`), 3. Enhance (`enhance`), 4. U-Net (`unet`), Data Samples, Deep Learning CT Reconstruction, Notes, Pipeline Modes (+1 more)

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (8): DATA_DIR, OUTPUTS_DIR, PYTHONPATH, REPO_ROOT, SAMPLE_1_DIR, SAMPLE_2_DIR, SRC_DIR, paths.sh script

### Community 16 - "Community 16"
Cohesion: 0.33
Nodes (6): create_conv_net(), crop_and_concat(), make_model(), Creates a new convolutional unet for the given parametrization.      :param x:, train(), train_step()

### Community 17 - "Community 17"
Cohesion: 0.25
Nodes (7): 1. Classical (`classical`), 2. Sinogram (`sinogram`), 3. U-Net (`unet`), Data Samples, Deep Learning CT Reconstruction, Notes, Pipeline Modes

### Community 18 - "Community 18"
Cohesion: 0.43
Nodes (4): make_model(), redcnn(), train(), train_step()

### Community 19 - "Community 19"
Cohesion: 0.67
Nodes (3): main(), scale_ascii_stl(), Path

### Community 28 - "Community 28"
Cohesion: 0.13
Nodes (14): 1. The Physics: How We Simulate a CT Scanner (`DATACREATION`), 2. The AI & ML Architecture (`DeepLearningCT`), 3. Codebase Architecture, 4. Execution Manual: How to Run the Pipeline, Classical Reconstruction (The Baseline), Core Concepts Simulated, Deep Learning Enhancement (U-Net), Digital Twin CT Pipeline & AI Reconstruction Manual (+6 more)

## Knowledge Gaps
- **54 isolated node(s):** `paths.sh script`, `REPO_ROOT`, `DATA_DIR`, `OUTPUTS_DIR`, `SRC_DIR` (+49 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `resolve_repo_path()` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 10`, `Community 11`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._
- **Why does `load_cto_settings()` connect `Community 1` to `Community 5`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `get_sample_config()` connect `Community 5` to `Community 1`?**
  _High betweenness centrality (0.044) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `DegradedProjectionData` (e.g. with `CTScanData` and `DegradedProjectionData`) actually correct?**
  _`DegradedProjectionData` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Creates a new convolutional unet for the given parametrization.      :param x:`, `Run FDK reconstruction for sample_2.`, `paths.sh script` to the rest of the system?**
  _71 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07058001397624039 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.13360323886639677 - nodes in this community are weakly interconnected._