
import vedo
import numpy as np

from vedo import Mesh

from acoustools.Mesh.Filename import scatterer_file_name
from acoustools.Mesh.Features import get_centre_of_mass_as_points


def scale_to_diameter(scatterer:Mesh , diameter: float, reset:bool=True, origin:bool=True) -> None:
    '''
    Scale a mesh to a given diameter in the x-axis and recomputes normals and areas \n
    Modifies scatterer in place so does not return anything.\n

    :param scatterer: The scatterer to scale
    :param diameter: The diameter target
    '''
    x1,x2,y1,y2,z1,z2 = scatterer.bounds()
    diameter_sphere = x2 - x1
    scatterer.scale(diameter/diameter_sphere,reset=reset, origin=origin)
    scatterer.compute_cell_size()
    scatterer.compute_normals()
    scatterer.filename = scatterer_file_name(scatterer)
    


def translate(scatterer:Mesh, dx:float=0,dy:float=0,dz:float=0) -> None:
    '''
    Translates a scatterer by (dx,dy,dz) \n
    Modifies inplace so does not return a value \n
    :param scatterer: The scatterer to use
    :param dx: Translation in the x direction
    :param dy: Translation in the y direction
    :param dz: Translation in the z direction
    '''
    scatterer.shift(np.array([dx,dy,dz]))
    scatterer.filename = scatterer_file_name(scatterer)

def rotate(scatterer:Mesh, axis:tuple[int], rot:float, centre:tuple[int]=(0, 0, 0), rotate_around_COM:bool=False):
    '''
    Rotates a scatterer in axis by rot\n
    Modifies inplace so does not return a value\n
    :param scatterer: The scatterer to use
    :param axis: The axis to rotate in
    :param rot: Angle to rotate in degrees
    :param centre: point to rotate around
    :param rotate_around_COM: If True will set `centre` to `scatterer`s centre of mass
    '''
    if rotate_around_COM:
        centre = vedo.vector(get_centre_of_mass_as_points(scatterer).cpu().detach().squeeze())

    if axis[0]:
        scatterer.metadata["rotX"] = scatterer.metadata["rotX"] + rot
    if axis[1]:
        scatterer.metadata["rotY"] = scatterer.metadata["rotY"] + rot
    if axis[2]:
        scatterer.metadata["rotZ"] = scatterer.metadata["rotZ"] + rot
    scatterer.rotate(rot, axis,point=centre)
    scatterer.filename = scatterer_file_name(scatterer)

 
def downsample(scatterer:Mesh, factor:int=2, n:int|None=None, method:str='quadric', boundaries:bool=False, compute_areas:bool=True, compute_normals:bool=True) -> Mesh:
    '''
    Downsamples a mesh to have `factor` less elements\n
    :param scatterer: The scatterer to use
    :param factor: The factor to downsample by
    :param n: The desired number of final points, passed to `Vedo.Mesh.decimate`
    :param method:, `boundaries` - passed to `vedo.decimate`
    :param compute_areas: if true will call `scatterer.compute_cell_size()`. Default `True`
    :param compute_normals: if true will call `scatterer.compute_normals()`. Default `True`
    :return: downsampled mesh
    '''
    scatterer_small =  scatterer.decimate(1/factor, n, method, boundaries)
    
    scatterer_small.metadata["rotX"] = scatterer.metadata["rotX"]
    scatterer_small.metadata["rotY"] = scatterer.metadata["rotY"]
    scatterer_small.metadata["rotZ"] = scatterer.metadata["rotZ"]

    if compute_areas: scatterer_small.compute_cell_size()
    if compute_normals: 
        scatterer_small.compute_normals()

    scatterer_small.filename = scatterer_file_name(scatterer_small)  + "-scale-" + str(factor)


    return scatterer_small


def centre_scatterer(scatterer:Mesh) -> list[int]:
    '''
    Translate scatterer so the centre of mass is at (0,0,0)\n
    Modifies Mesh in place \n
    :param scatterer: Scatterer to centre
    :return: Returns the amount needed to move in each direction
    '''
    com = get_centre_of_mass_as_points(scatterer).cpu()
    correction = [-1*com[:,0].item(), -1*com[:,1].item(), -1*com[:,2].item()]
    translate(scatterer, dx = correction[0], dy = correction[1], dz=  correction[2])

    return correction

