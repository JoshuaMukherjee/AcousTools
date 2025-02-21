from acoustools.Mesh import load_scatterer, cut_mesh_to_walls, scale_to_diameter, get_normals_as_points, get_centres_as_points
from acoustools.Visualiser import Visualise_mesh, force_quiver_3d
import vedo

path = "../BEMMedia"
msh = "/Teapot.stl"


scatterer = load_scatterer(msh, root_path=path)
scale_to_diameter(scatterer, 0.1)
# vedo.show(scatterer)

cut = cut_mesh_to_walls(scatterer, layer_z=0.008253261256963015, wall_thickness=0.001)

normals = get_normals_as_points(cut)
centres = get_centres_as_points(cut)

force_quiver_3d(centres,normals[:,0],normals[:,1],normals[:,2], scale=1e-4)
# vedo.show(normals)
# Visualise_mesh(cut,equalise_axis=True)