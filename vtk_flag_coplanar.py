  #!python
# flag cells whose value is the same as all its neighbors
# coplanar_borders: points on border also can be coplanar

'''
usage: $0 data*vtk,csv values:data flag=coplanar coplanar_borders@ output*vtk,csv diplay@
'''
import sys, os.path, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, log

from pd_vtk import vtk_Voxel, pv_save, vtk_mesh_info, vtk_array_ijk, vtk_reshape_a3d

def vtk_flag_coplanar(data, values, flag, coplanar_borders, output, display):
  grid = vtk_Voxel.factory(data)
  if grid is None:
    print("data is not a schema or a grid")
    return 1
  grid.cell_data[flag] = grid.coplanar(values, int(coplanar_borders))
  print(vtk_mesh_info(grid))
  if output:
    pv_save(grid, output)

  if int(display):
    from db_voxel_view import pd_voxel_view
    #pd_voxel_view(vtk_array_ijk(grid, flag), None, flag)
    pd_voxel_view(vtk_reshape_a3d(grid.dimensions, grid.get_array(flag), True), None, flag)

  log("# vtk_flag_coplanar finished")

main = vtk_flag_coplanar

if __name__=="__main__":
  usage_gui(__doc__)
