import astra
import numpy as np
vol_geom = astra.create_vol_geom(10, 10, 10)
angles = np.deg2rad(np.arange(0, 5))
proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0, 20, 20, angles)
voxels = np.zeros((10, 10, 10), dtype=np.float32)
proj_id = astra.creators.create_sino3d_gpu(voxels, proj_geom, vol_geom, returnData=False)
data = astra.data3d.get(proj_id)
print("Data shape:", data.shape)
