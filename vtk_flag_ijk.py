#!python
# flag ijk. only works on vtk grid objects.
# v1.0 2022/05 paulo.ernesto
'''
usage: $0 block_model*vtk,csv,xlsx flag_var=vtk_ijk preference%cell,point output*vtk display@
'''
import sys, os.path
import numpy as np
import pandas as pd

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')
from _gui import usage_gui, log

from pd_vtk import pv_read, pv_save, vtk_plot_meshes, vtk_grid_flag_ijk

def vtk_flag_ijk(points, flag_var, preference, output, display):
  grid = pv_read(points)

  vtk_grid_flag_ijk(grid, flag_var, preference)

  if output:
    pv_save(grid, output)
    
  if int(display):
    vtk_plot_meshes(grid, scalars=flag_var)

main = vtk_flag_ijk

if __name__=="__main__":
  usage_gui(__doc__)
