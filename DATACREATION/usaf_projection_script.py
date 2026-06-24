import os
import zipfile
import numpy as np
import astra
import trimesh
import gc
import argparse
from skimage.io import imsave
from skimage.measure import block_reduce

# Set this to True to simulate real-world scanner physics (photon starvation and detector noise).
# Set to False to output mathematically perfect anti-aliased projections.
ADD_PHYSICS_NOISE = True

def run_local_projection_pipeline(stl_filepath, output_dir="projection_slices_tiff", zip_filename="usaf_phantom_projections.zip", supersample_pitch=0.025, downsample_factor=2, add_noise=True, i0=50000.0, gaussian_std=10.0):
    # --- STEP 1: LOCATE YOUR AUTOCAD STL FILE ---
    if not os.path.exists(stl_filepath):
        raise FileNotFoundError(f"Error: Could not find '{stl_filepath}'. Please check the path.")
    print(f"Successfully located: {stl_filepath}")

    # --- STEP 2: SUPERSAMPLED VOXELIZATION ---
    print("Loading geometry mesh...")
    mesh = trimesh.load_mesh(stl_filepath)

    print("Auto-aligning mesh orientation...")
    extents = mesh.extents
    min_dim_idx = np.argmin(extents)
    if min_dim_idx == 0:
        print("  -> Shortest dimension is X. Rotating 90 degrees around Y to stand cylinder upright.")
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    elif min_dim_idx == 1:
        print("  -> Shortest dimension is Y. Rotating 90 degrees around X to stand cylinder upright.")
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    else:
        print("  -> Cylinder is already upright (shortest dimension is Z).")

    print(f"Voxelizing 3D geometry... (Super-sampling at pitch={supersample_pitch} for anti-aliasing)")
    # Voxelize at specified pitch.
    # With 400GB RAM, we can afford this massive intermediate matrix.
    voxel_obj = mesh.voxelized(pitch=supersample_pitch, max_iter=1000)
    voxels_high_res = voxel_obj.matrix.astype(np.float32)
    
    # Delete mesh to free RAM
    del mesh
    gc.collect()

    print(f"Downsampling with factor {downsample_factor} to generate smooth partial-volume edges...")
    # Pad to make dimensions divisible by downsample_factor
    pad_x = (downsample_factor - voxels_high_res.shape[0] % downsample_factor) % downsample_factor
    pad_y = (downsample_factor - voxels_high_res.shape[1] % downsample_factor) % downsample_factor
    pad_z = (downsample_factor - voxels_high_res.shape[2] % downsample_factor) % downsample_factor
    if pad_x or pad_y or pad_z:
        voxels_high_res = np.pad(voxels_high_res, ((0, pad_x), (0, pad_y), (0, pad_z)), mode='constant')
        
    # Average-pool downsample to get greyscale partial-volume voxels (anti-aliasing)
    voxels = block_reduce(voxels_high_res, block_size=(downsample_factor, downsample_factor, downsample_factor), func=np.mean)
    
    del voxels_high_res
    gc.collect()

    x_dim, y_dim, z_dim = voxels.shape
    print(f"Final 50-micron volume matrix dimensions locked at: {voxels.shape}")

    # --- STEP 3: CONFIGURE ASTRA CT SCAN GEOMETRY ---
    angles = np.deg2rad(np.arange(0, 360, 1))

    # Setup 3D Volume Geometry
    vol_geom = astra.creators.create_vol_geom(y_dim, z_dim, x_dim)

    # Setup 3D Cone Beam Detector Geometry
    # To mirror physical hardware and prevent "horizontal strip" aspect ratios,
    # we simulate a perfectly square detector that covers the entire bounding box diagonally.
    max_diagonal = int(np.ceil(np.sqrt(x_dim**2 + y_dim**2))) + 20
    detector_size = max(max_diagonal, z_dim + 20)
    if detector_size % 2 != 0:
        detector_size += 1  # Ensure even number of pixels for symmetric center of rotation
        
    detector_cols = detector_size
    detector_rows = detector_size 
    
    voxel_size_mm = supersample_pitch * downsample_factor
    sod_mm = 160.0
    sdd_mm = 200.0
    det_pixel_mm = 0.048
    
    sod_voxels = sod_mm / voxel_size_mm
    sdd_voxels = sdd_mm / voxel_size_mm
    det_spacing = det_pixel_mm / voxel_size_mm

    proj_geom = astra.create_proj_geom(
        'cone', 
        det_spacing, det_spacing, 
        detector_rows, detector_cols, 
        angles, 
        sod_voxels, sdd_voxels - sod_voxels
    )

    # --- STEP 4: RUN FORWARD PROJECTION ON THE GPU ---
    print("Simulating X-ray attenuation projection sweeps on GPU...")
    # Generate projections and keep them in ASTRA's GPU memory
    proj_id = astra.creators.create_sino3d_gpu(voxels, proj_geom, vol_geom, returnData=False)
    
    del voxels
    gc.collect()

    # --- STEP 5: NOISE INJECTION & HIGH-PRECISION EXPORT ---
    projections_dir = os.path.join(output_dir, "projections")
    os.makedirs(projections_dir, exist_ok=True)
    print(f"Fetching 3D sinogram from GPU...")

    # Fetch the full 3D sinogram from ASTRA memory
    projections = astra.data3d.get(proj_id)
    astra.data3d.delete(proj_id)
    gc.collect()

    if add_noise:
        print("Injecting realistic physics noise (Poisson & Gaussian)...")
        transmission = i0 * np.exp(-projections)
        
        # Poisson noise (photon statistics)
        noisy_transmission = np.random.poisson(transmission)
        # Gaussian noise (detector electronics)
        noisy_transmission = noisy_transmission + np.random.normal(0, gaussian_std, noisy_transmission.shape)
        
        # Prevent log of <= 0 by clipping to 1 photon
        noisy_transmission = np.clip(noisy_transmission, 1, None)
        projections = -np.log(noisy_transmission / i0)

    # Calculate global max for 16-bit normalization
    sino_max = np.max(projections)
    if sino_max <= 0: sino_max = 1.0
    
    print(f"Writing 360 16-bit TIFF files to local directory: '{output_dir}'...")

    for i in range(len(angles)):
        # Extract the 2D detector slice
        slice_2d = projections[:, i, :]
        upright_slice = np.flipud(slice_2d)
        
        # Normalize and convert to 16-bit for high-dynamic range saving without clipping
        normalized_slice = upright_slice / sino_max
        final_image = (normalized_slice * 65535).astype(np.uint16)

        filename = os.path.join(projections_dir, f"projection_angle_{i:03d}.tiff")
        imsave(filename, final_image, check_contrast=False)

    print("All TIFF frames generated successfully!")
    
    # Generate settings.cto for DeepLearningCT compatibility
    settings_content = f"""[Device settings]
mA = 1.000000
kV = 35.000000

[Detector settings]
binning = 0
frames = 1
exp time (ms) = 500.000000
CORunbinned = {detector_cols / 2}
pixel size = {det_pixel_mm}
Xmin = 0
Xmax = {detector_cols}
Ymin = 0
Ymax = {detector_rows}
VC = {detector_rows / 2}

[CT scan settings]
projections = {len(angles)}
angle range = 360.000000
CWCCW = FALSE
SOD = {sod_mm}
SDD = {sdd_mm}

[CT reconstruction settings]
SOD = {sod_mm}    
SDD = {sdd_mm}    
COR = {detector_cols / 2}    
vertical center = {detector_rows / 2}    
last angle = 360.000000    
bhc = 0.000000    
filter strength = 0.000000    
projections = {len(angles)}    
rows = {detector_rows}    
columns = {detector_cols}    
pixel_size (mm) = {det_pixel_mm}    
zmax = {detector_rows - 1}    
zmin = 0    
direction = 1    
tilt = 0.000000    
xminrec = 0    
xmaxrec = {detector_cols}    
yminrec = 0    
ymaxrec = {detector_rows}    
interpolate = FALSE    
"""
    cto_path = os.path.join(output_dir, "settings.cto")
    with open(cto_path, "w") as f:
        f.write(settings_content)
    
    print(f"settings.cto written to {cto_path}")

    # --- STEP 6: ZIP THE COMPRESSED OUTPUT ---
    print(f"Compressing frames into '{zip_filename}' in your current working directory...")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files_list in os.walk(output_dir):
            for file in files_list:
                zipf.write(os.path.join(root, file), arcname=file)

    print(f"Process complete! Output saved locally as {zip_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CT projections from an STL phantom.")
    parser.add_argument("--stl", type=str, default="STL/FINAL30.stl", help="Path to the input STL file.")
    parser.add_argument("--output_dir", type=str, default="projection_slices_tiff", help="Directory to save output TIFF projections.")
    parser.add_argument("--zip_name", type=str, default="usaf_phantom_projections.zip", help="Name of the output zip file.")
    parser.add_argument("--supersample", type=float, default=0.025, help="Pitch for supersampled voxelization.")
    parser.add_argument("--downsample", type=int, default=2, help="Block reduction factor for downsampling.")
    parser.add_argument("--no-noise", action="store_true", help="Disable physics noise injection.")
    parser.add_argument("--i0", type=float, default=50000.0, help="Initial photon count for Poisson noise.")
    parser.add_argument("--gaussian-std", type=float, default=10.0, help="Standard deviation for Gaussian detector noise.")
    
    args = parser.parse_args()
    
    # ADD_PHYSICS_NOISE is True by default in the script. If --no-noise is passed, we set add_noise=False.
    add_noise = not args.no_noise and ADD_PHYSICS_NOISE
    
    run_local_projection_pipeline(
        stl_filepath=args.stl,
        output_dir=args.output_dir,
        zip_filename=args.zip_name,
        supersample_pitch=args.supersample,
        downsample_factor=args.downsample,
        add_noise=add_noise,
        i0=args.i0,
        gaussian_std=args.gaussian_std
    )