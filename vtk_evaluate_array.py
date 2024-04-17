#!python
# evaluate code interpolating vtk array names as symbols
# optionally, export a array to as a 3d tiff raster
# v1.0 2023/08 paulo.ernesto
'''
usage: $0 input_path*vtk input_code:input_path output*vtk,tif
'''

import sys, os.path, re
import numpy as np
import pandas as pd

# import modules from a pyz (zip) file with same name as scripts
sys.path.insert(0, os.path.splitext(sys.argv[0])[0] + '.pyz')

from _gui import usage_gui

class DSL_(object):
  def __call__(self, code = ''):
    if not code:
      return
    elif code == '-':
      code = input()
    elif code.endswith('.py') and os.path.exists(code):
      code = open(code).read()


    s = re.sub(r'([a-z_]\w{1,})(\s+=\s+)?', self.sub, code)
    print(s)
    exec(s)

  def sub(self, m):
    token = m.group(1)
    if (m.group(2) is not None) or (token in self.array_names):
      token = "self.cell_data['%s']%s" % m.groups('')
    return token

  @classmethod
  def factory(cls, fp):
    import pyvista as pv
    self = None
    if fp.lower().endswith('tif'):
      from pd_vtk import vtk_tif_to_grid
      self = vtk_tif_to_grid(fp)
    else:
      self = pv.read(fp)
    f = None
    if self.GetDataObjectType() == 6:
      f = type('DSLUG', (DSL_, pv.ImageData), {})
    elif self.GetDataObjectType() == 2:
      f = type('DSLSG', (DSL_, pv.StructuredGrid), {})
    return f(self)

# main
def vtk_evaluate_array(input_path, input_code, output = None):
  dsl = DSL_.factory(input_path)
  if not output:
    output = input_path
  dsl(input_code)
  if output.lower().endswith('tif'):
    from pd_vtk import vtk_grid_array_to_tif
    vtk_grid_array_to_tif(dsl, input_code, output)
  elif output.lower().endswith('csv'):
    from pd_vtk import vtk_mesh_to_df
    df = vtk_mesh_to_df(dsl)
    df.to_csv(output, index=False)
  else:
    dsl.save(output)

main = vtk_evaluate_array

if __name__=="__main__": 
  usage_gui(__doc__)
