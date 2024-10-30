import vedo, math, torch
from vedo import Mesh

from acoustools.Mesh import translate, get_centres_as_points
from acoustools.Paths import interpolate_path, target_distance_to_n, total_distance

def slice_mesh(mesh:Mesh, dz:float=0.0005, set_to_floor:bool=True) -> list[Mesh]:
    '''
    Wont deal well with overhangs
    '''
    xmin,xmax, ymin,ymax, zmin,zmax = mesh.bounds()
    if set_to_floor:
        translate(mesh, dz=-1*zmin)
    
    xmin,xmax, ymin,ymax, zmin,height = mesh.bounds()

    num_layers = math.ceil(height/ dz)

    origin = [0,0,0]
    norm = (0,0,1)

    layers = []
    c = 1
    for i in range(num_layers):
        layer = mesh.clone().intersect_with_plane(origin, norm)
        origin[2] += dz

        boundary = layer.boundaries()
        edges = boundary.split()
        layers+=(edges)
    
    return layers
