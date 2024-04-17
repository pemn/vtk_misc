#!python
# flag points by 2d polygon attributes
# v1.1 2024/03 paulo.ernesto
# v1.0 2023/12 paulo.ernesto
'''
usage: $0 points*vtk,csv,xlsx shape*shp var:shape mode%cell,point output*vtk,csv,xlsx
'''
import sys, os.path, re
import numpy as np
import pandas as pd
from shapely import Point, Polygon

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, pd_load_dataframe, pd_save_dataframe, pd_detect_xyz, log

import pyvista as pv
from pd_vtk import pv_read, vtk_Voxel, vtk_mesh_info, vtk_df_to_meshes, vtk_grid_points_to_df

def vtk_flag_by_shape(points, shape, var, mode, output):
  df = pd_load_dataframe(shape)
  xyz = pd_detect_xyz(df)
  meshes = vtk_df_to_meshes(df, xyz, var)

  if re.fullmatch(r'[\d\.\-,;_~]+', points):
    bb = [df[xyz].min().values, df[xyz].max().values]
    #print(bb)
    grid = vtk_Voxel.from_bb_schema(bb, points, 2)
    grid.cells_volume('volume')
  else:
    grid = pv_read(points)

  if mode == 'cell' and grid.n_cells == 0:
    mode = 'point'
  
  vs = None
  ps = None
  if mode == 'point':
    pl = [Point(*_) for _ in grid.points]
    vs = np.empty(grid.n_points, dtype=df[var].dtype)
  else:
    pl = [Point(*_) for _ in grid.cell_centers().points]
    vs = np.empty(grid.n_cells, dtype=df[var].dtype)

  for mesh in meshes:
    if len(mesh.points) < 3:
      continue
    poly = Polygon(mesh.points)
    # take the most common value as a scalar
    data = mesh.point_data[var]
    data_major = pd.Series(data).value_counts().idxmax()

    for i,p in enumerate(pl):
      if poly.contains(p):
        vs[i] = data_major

  if mode == 'point':
    grid.point_data[var] = vs
  else:
    grid.cell_data[var] = vs

  print(vtk_mesh_info(grid))
  if output.lower().endswith('vtk'):
    grid.save(output)
  else:
    pd_save_dataframe(vtk_grid_points_to_df(grid, xyz), output)


main = vtk_flag_by_shape

if __name__=="__main__":
  usage_gui(__doc__)
