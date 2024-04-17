#!python
# multiple utilities for vtk format files
# v1.0 01/2022 paulo.ernesto
'''
usage: $0 input_path*vtk,obj,msh,00t var:input_path info@ extract_largest@ footprint@ bounding_box@ factorize@ ctp@ ptc@ delete_array@ set_active@ output*vtk,obj,msh,00t display@
'''
import sys, os.path
import pandas as pd
import numpy as np

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui, commalist, log

import pyvista as pv

def vtk_util(input_path, var, info, extract_largest, footprint, bounding_box, factorize, ctp, ptc, delete_array, set_active, output, display):
  from pd_vtk import pv_read, pv_save, vtk_plot_meshes, vtk_mesh_info, vtk_meshes_bb, vtk_Voxel
  mesh = pv_read(input_path)
  mesh_ori = mesh

  if int(info):
    vtk_mesh_info(mesh)
  
  if var:
    if var in mesh.array_names and var != mesh.active_scalars_name:
      arr = mesh.get_array(var)
      if arr.dtype.num < 17:
        mesh.set_active_scalars(var)
  
  if int(extract_largest):
    mesh = mesh.threshold(0.5).connectivity(True)

  if int(footprint):
    bb = vtk_meshes_bb(mesh)
    grid = vtk_Voxel.from_bb(bb, 10, 2)

    cutoff = 0.01
    
    if not var:
      var = 'footprint'

    if var not in mesh.array_names:
      if sys.hexversion < 0x3080000:
        mesh.point_arrays[var] = np.ones(mesh.n_points)
      else:
        mesh.point_data[var] = np.ones(mesh.n_points)
    
    mesh.set_active_scalars(var)

    cutoff = np.nanmean(mesh.active_scalars)

    mesh = grid.interpolate(mesh.ctp(), strategy='closest_point')
    mesh.set_active_scalars(var)

    # masks the edges of the grid so the contour will pass there
    i2d = np.zeros((grid.dimensions[1], grid.dimensions[0]), dtype='bool')
    i2d[0, :] = True
    i2d[:, 0] = True
    i2d[-1, :] = True
    i2d[:, -1] = True

    np.place(mesh.active_scalars, i2d.flat, 0)

    mesh = mesh.contour([cutoff], preference='cell')

  if int(extract_largest):
    mesh = mesh.extract_surface()
  if int(bounding_box):
    mesh = mesh.outline()
  if int(factorize) and var:
    var_i = f'{var}_i'
    u, n = np.unique(mesh.get_array(var), return_inverse=True)
    if mesh.get_array_association(var) == pv.FieldAssociation.CELL:
      mesh.cell_data[var_i] = n
    else:
      mesh.point_data[var_i] = n
  
  if int(ptc):
    mesh = mesh.ptc()
  if int(ctp):
    mesh = mesh.ctp()

  if int(delete_array):
    if mesh.get_array_association(var) == pv.FieldAssociation.CELL:
      del mesh.cell_data[var]
    else:
      del mesh.point_data[var]
  
  if int(set_active):
    mesh.set_active_scalars(var or None)

  print(vtk_mesh_info(mesh))
  if output:
    pv_save(mesh, output)

  if int(display):
    vtk_plot_meshes([mesh,mesh_ori])

  log("# finished")

main = vtk_util

if __name__=="__main__":
  usage_gui(__doc__)
