
from acoustools.Utilities import device, DTYPE, BOARD_POSITIONS
import acoustools.Constants as Constants

import vedo, torch
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from vedo import Mesh

from acoustools.Mesh.Load import load_scatterer
from acoustools.Mesh.Transform import centre_scatterer, scale_to_diameter
from acoustools.Mesh.Features import get_centres_as_points, get_normals_as_points


def mesh_to_board(path:str, compute_areas:bool = True, compute_normals:bool=True, dx:float=0,
                   dy:float=0,dz:float=0, rotx:float=0, roty:float=0, rotz:float=0, root_path:str="", force:bool=False, flip_normals = True, diameter = 2*BOARD_POSITIONS, centre=True):
    '''
    Loads a scatterer as a `vedo` `Mesh` and interprets it as a transducer board with a transducer at each mesh centre
    :param path: The name of the scatterer to load
    :param compute_areas: if `True` will call `scatterer.compute_cell_size()`. Default `True`
    :param compute_normals: if `True` will call `scatterer.compute_normals()`. Default `True`
    :param dx: Translation in the x direction to apply
    :param dy: Translation in the y direction to apply
    :param dz: Translation in the z direction to apply
    :param rotx: Rotation around the x axis to apply
    :param roty: Rotation around the y axis to apply
    :param rotz: Rotation around the z axis to apply
    :param root_path: The folder containing the file, the scatterer to be loaded will be loaded from `root_path+path`
    :param flip_normals: If True will flip the normals 
    :param diameter: Size to scale the mesh to in thr x-axis
    :return: The `vedo` `Mesh` of the scatterer
    '''
    
    scatterer = load_scatterer(path=path,compute_areas=compute_areas,  compute_normals=compute_normals, 
                               dx=dx, dy=dy, dz=dz, rotx=rotx, roty=roty, rotz=rotz, root_path=root_path,force=force)
    
    if centre: centre_scatterer(scatterer)
    
    if diameter is not None: scale_to_diameter(scatterer, diameter)
    
    centres = get_centres_as_points(scatterer).squeeze(0).permute(1,0)
    norms = get_normals_as_points(scatterer).squeeze(0).permute(1,0)  
    if flip_normals: norms = norms * -1

    return centres, norms
    