


import vedo

from vedo import Mesh

from acoustools.Mesh.Load import calculate_features, scatterer_file_name
from acoustools.Mesh.Transform import translate
from acoustools.Mesh.Features import get_centre_of_mass_as_points 


def cut_mesh_to_walls(scatterer:Mesh, layer_z:float, layer_normal:tuple[float] = (0,0,-1.0), wall_thickness = 0.001) -> Mesh:
    '''
    Cuts a mesh with a given plane and then converts the result to have walls of a certain thickness \n
    :param scatterer: Mesh to use
    :param layer_z: coordinate of layer
    :param layer_normal: Normal to layer (if not +- (0,0,1) then layer_z will not refer to a z coordinate)
    :param wall_thickness: Thickness of the walls to returns
    :return: Cut mesh with walls
    '''

    xmin,xmax, ymin,ymax, zmin,zmax = scatterer.bounds()
    dx = xmax-xmin
    dy = ymax-ymin

    scale_x = (dx-2*wall_thickness) / dx
    scale_y = (dy-2*wall_thickness) / dy

    outler_layer = scatterer.cut_with_plane((0,0,layer_z),layer_normal)
    inner_layer = outler_layer.clone()
    inner_layer.scale((scale_x,scale_y,1), origin=False)

    com_outer = get_centre_of_mass_as_points(outler_layer)
    com_inner = get_centre_of_mass_as_points(inner_layer)

    d_com = (com_outer - com_inner).squeeze()

    translate(inner_layer, *d_com)

    walls = vedo.merge(outler_layer, inner_layer)


    boundaries_outer = outler_layer.boundaries()
    boundaries_inner = inner_layer.boundaries()

    strips = boundaries_outer.join_with_strips(boundaries_inner).triangulate()
    

    walls = vedo.merge(walls,strips)
    
    calculate_features(walls)
    scatterer_file_name(walls)

    return walls.clean()

def cut_closed_scatterer(scatterer:Mesh,layer_z:float, normals=[(0,0,1)]):
    '''
    Cuts a scatterer across a z-plane\\
    :param scatterer: Mesh
    :param layer_z: height to cute
    :param normals: Which way is up 
    '''
    origins=[(0,0,layer_z)]
    closed_scatterer = scatterer.cut_closed_surface(origins=origins, normals=normals)
    return closed_scatterer