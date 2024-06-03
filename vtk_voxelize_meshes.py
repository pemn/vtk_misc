#!python
# repair solid meshes by voxelization
# similar to pyvista.voxelize but with regular cell size
# v1.0 2022/02 paulo.ernesto
'''
usage: $0 input_files#input_path*vtk,obj,msh cell_size=1,5,10,50 tolerance=0.0,0.001 output_suffix=g display@
'''
'''
Copyright 2024 Vale

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*** You can contribute to the main repository at: ***

https://github.com/pemn/vtk_util
---------------------------------
'''
import sys, os.path
import numpy as np
import pandas as pd
import pyvista as pv
from glob import glob

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, commalist, log

from pd_vtk import pv_read, pv_save, vtk_Voxel, vtk_plot_meshes

def vtk_voxelize_mesh(mesh, cell_size = 10, tolerance = 0.001):
  grid = vtk_Voxel.from_mesh(mesh, cell_size)
  mask = grid.select_enclosed_points(mesh, tolerance, check_surface=False)
  mask = mask.point_data['SelectedPoints'].view(np.bool_)
  mask = grid.extract_points(mask)
  del mask.cell_data['vtkOriginalCellIds']
  del mask.point_data['vtkOriginalPointIds']
  return mask.extract_surface(False, False)

def vtk_voxelize_meshes(meshes, cell_size, tolerance, suffix, display):
  if not cell_size:
    cell_size = 10
  else:
    cell_size = float(cell_size)

  if not tolerance:
    tolerance = 0.001
  else:
    tolerance = float(tolerance)

  if not suffix:
    suffix = 'g'
  r = []

  for fp in commalist(meshes).split():
    if not os.path.exists(fp):
      log('file not found:', fp)
      continue
    mesh = vtk_voxelize_mesh(pv_read(fp), cell_size, tolerance)
    r.append(mesh)
    se = os.path.splitext(fp)
    output = '%s_%s%d%s' % (se[0], suffix, cell_size, se[1])
    log(fp,'➔',output)
    mesh.save(output)

  if int(display):
    vtk_plot_meshes(r)

main = vtk_voxelize_meshes

if __name__=="__main__":
  usage_gui(__doc__)
