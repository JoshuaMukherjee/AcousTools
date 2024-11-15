from acoustools.Mesh import load_scatterer, cut_mesh_to_walls, scale_to_diameter
from acoustools.Visualiser import Visualise_mesh
import vedo

path = "../BEMMedia"
msh = "/Teapot.stl"


scatterer = load_scatterer(msh, root_path=path)
scale_to_diameter(scatterer, 0.1)
# vedo.show(scatterer)

cut = cut_mesh_to_walls(scatterer, layer_z=0.008253261256963015, wall_thickness=0.001)
Visualise_mesh(cut,equalise_axis=True)