from acoustools.Mesh import load_scatterer, get_edge_data
from acoustools.Visualiser import Visualise_mesh
import vedo
import numpy as np

path = '../BEMMedia/'
scatterer = load_scatterer('Hand-0-lam2.STL', root_path=path).fill_holes(100)
scatterer.subdivide(1)
scatterer.clean()

get_edge_data(scatterer)

