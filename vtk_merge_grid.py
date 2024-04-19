#!python
# transfer data between VTK grids with different schemas (reblock)

'''
usage: $0 source_grid*vtk target_grid*vtk output*vtk display@1 variable:source_grid
'''
import sys, os.path, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, log
import pyvista as pv

from pd_vtk import vtk_shape_ijk, vtk_array_ijk, vtk_mesh_info, vtk_spacing_fit, vtk_cell_size, vtk_plot_grids

def major(_):
  return pd.Series.value_counts(_).idxmax()

def axis_grid(a, index):
  print(a)
  print(index)
  return np.sum(a)

def vtk_merge_grid(grid0, grid1):
  c0 = vtk_cell_size(grid0)
  c1 = vtk_cell_size(grid1)

  if np.greater(c0, c1).any():
    grid0 = vtk_break_grid(grid0, grid1)

  if np.greater(c0, c1).all():
    grid1 = grid0
  else:
    grid1 = vtk_scale_grid(grid0, grid1)

  return grid1

def vtk_scale_grid(grid0, grid1):
  for name in grid0.array_names:
    cell = grid0.get_array_association(name) == pv.FieldAssociation.CELL
    s0 = vtk_array_ijk(grid0, None, cell)
    s1 = vtk_array_ijk(grid1, None, cell)
    d0 = grid0.get_array(name)
    f = major
    # check if dtype is some kind of float
    if d0.dtype.num in (11, 12, 13, 23):
      if np.nanmax(d0) > 100:
        f = np.nansum
      else:
        f = np.nanmean

    log(name, d0.dtype, f.__name__, '⊞', s0.shape, np.prod(s0.shape), '➔', s1.shape, np.prod(s1.shape))
    d1 = scale_array(s0, s1, d0, f)
    if cell:
      grid1.cell_data[name] = d1
    else:
      grid1.point_data[name] = d1

  return grid1


def scale_array(s0, s1, d0, f):
  n01 = np.prod(s0.shape) // np.prod(s1.shape)
  b01s = (n01,) + s1.shape
  t0 = np.reshape(np.moveaxis(np.broadcast_to(s1, b01s), 0, -1), (np.prod(s1.shape), n01))
  d1 = np.empty(t0.shape[0], dtype=d0.dtype)
  for i in range(t0.shape[0]):
    d1[i] = f(np.take(d0, t0[i]))
  return d1


def vtk_break_grid(grid0, grid1):
  d0 = vtk_shape_ijk(grid0.dimensions)
  d1 = vtk_shape_ijk(grid1.dimensions)
  t0 = np.minimum(d0, d1)
  t1 = np.maximum(d0, d1)
  #log('d0 =',*d0,'| d1 =',*d1)
  #log('t0 =',*t0,'| t1 =',*t1)

  spacing = vtk_spacing_fit(grid1.dimensions, grid1.spacing, np.flip(t1))
  grid = pv.ImageData(dimensions=np.flip(t1), spacing=spacing, origin=grid1.origin)

  for name in grid0.array_names:
    data = None
    s0 = None
    s1 = None
    if grid0.get_array_association(name) == pv.FieldAssociation.CELL:
      s0 = np.reshape(grid0.get_array(name), np.maximum(np.subtract(d0, 1), 1))
      s1 = np.empty(np.maximum(np.subtract(t1, 1), 1), dtype=s0.dtype)
      data = grid.cell_data
    else:
      s0 = np.reshape(grid0.get_array(name), d0)
      s1 = np.empty(t1, dtype=s0.dtype)
      data = grid.point_data
    log(name, s0.dtype, '⊞', s0.shape, np.prod(s0.shape), '➔', s1.shape, np.prod(s1.shape))
    mg = np.transpose(np.meshgrid(*[np.linspace(0, s0.shape[_], s1.shape[_], False, dtype=np.int_) for _ in range(len(d0))]), (2,1,3,0))
    it = np.nditer(s1, ['multi_index'])
    while not it.finished:
      s1[it.multi_index] = s0[tuple(mg[it.multi_index])]
      it.iternext()
    data[name] = s1.flat

  return grid

def main(source_grid, target_grid, output, display, variable):
  grid0 = pv.read(source_grid)
  grid1 = None
  if re.fullmatch(r'[\d\.\-,;_~]+', target_grid):
    dims = None
    spacing = np.resize(np.asfarray(re.split('[,_]', target_grid)), 3)
    origin = None
    if grid0.GetDataObjectType() == 2:
      b0 = np.reshape(grid0.bounds, (3,2))
      p0 = np.subtract(b0[:, 1], b0[:, 0])
      c0 = np.divide(p0, np.maximum(np.subtract(grid0.dimensions, 1), 1))
      dims = np.maximum(np.ceil(np.divide(np.add(p0, c0), spacing)), 1).astype(np.int_)
      origin = b0[:, 0]
    else:
      dims = np.maximum(np.ceil(np.divide(np.multiply(grid0.dimensions, grid0.spacing), spacing)), 1).astype(np.int_)
      origin = grid0.origin
    grid1 = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
  else:
    grid1 = pv.read(target_grid)

  grid = vtk_merge_grid(grid0, grid1)
  grid.set_active_scalars(grid.active_scalars_name)

  print(vtk_mesh_info(grid))
  if grid is not None and output:
    grid.save(output)

  if int(display):
    vtk_plot_grids([grid0, grid], variable)

if __name__=="__main__":
  usage_gui(__doc__)
