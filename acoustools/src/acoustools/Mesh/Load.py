

import vedo

from vedo import Mesh

from acoustools.Mesh.Transform import rotate, translate
from acoustools.Mesh.Filename import scatterer_file_name

def load_scatterer(path:str, compute_areas:bool = True, compute_normals:bool=True, dx:float=0,
                   dy:float=0,dz:float=0, rotx:float=0, roty:float=0, rotz:float=0, root_path:str="", force:bool=False, flip_normals=False) -> Mesh:
    '''
    Loads a scatterer as a `vedo` `Mesh` and applies translations as needed
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
    :return: The `vedo` `Mesh` of the scatterer
    '''
    scatterer = vedo.load(root_path+path, force=force)
    
    if scatterer is not None:
        if compute_areas: scatterer.compute_cell_size()
        if compute_normals: 
            scatterer.compute_normals()
            if flip_normals: scatterer.flip_normals()

        scatterer.metadata["rotX"] = 0
        scatterer.metadata["rotY"] = 0
        scatterer.metadata["rotZ"] = 0

        # scatterer.filename = scatterer.filename.split("/")[-1]
        scatterer.filename = scatterer_file_name(scatterer)

        scatterer.metadata["FILE"] = scatterer.filename.split(".")[0]


        rotate(scatterer,(1,0,0),rotx)
        rotate(scatterer,(0,1,0),roty)
        rotate(scatterer,(0,0,1),rotz)

        translate(scatterer,dx,dy,dz)
    else:
        raise ValueError(f"File not found at {path} - please check the path")

    return scatterer



def load_multiple_scatterers(paths:list[str],  compute_areas:bool = True, compute_normals:bool=True, 
                             dxs:list[int]=[],dys:list[int]=[],dzs:list[int]=[], rotxs:list[int]=[], rotys:list[int]=[], rotzs:list[int]=[], root_path:str="") -> Mesh:
    '''
    Loads multiple scatterers and combines them into a single scatterer object
    :param path: The name of the scatterers to load
    :param compute_areas: if true will call `scatterer.compute_cell_size()`. Default True
    :param compute_normals: if true will call `scatterer.compute_normals()`. Default True
    :param dxs: List of translations in the x direction to apply to each scatterer
    :param dys: List of translations in the y direction to apply to each scatterer
    :param dzs: List of translations in the z direction to apply to each scatterer
    :param rotxs: List pf rotations around the x axis to apply to each scatterer
    :param rotys: List pf rotations around the y axis to apply to each scatterer
    :param rotzs: List pf rotations around the z axis to apply to each scatterer
    :param root_path: The folder containing the file, the scatterer to be loaded will be loaded from `root_path+path`
    :return: A merged mesh from all of the paths provided
    '''
    dxs += [0] * (len(paths) - len(dxs))
    dys += [0] * (len(paths) - len(dys))
    dzs += [0] * (len(paths) - len(dzs))

    rotxs += [0] * (len(paths) - len(rotxs))
    rotys += [0] * (len(paths) - len(rotys))
    rotzs += [0] * (len(paths) - len(rotzs))

    scatterers = []
    for i,path in enumerate(paths):
        scatterer = load_scatterer(path, compute_areas, compute_normals, dxs[i],dys[i],dzs[i],rotxs[i],rotys[i],rotzs[i],root_path)
        scatterers.append(scatterer)
    combined = merge_scatterers(*scatterers)
    return combined

def merge_scatterers(*scatterers:Mesh, flag:bool=False) ->Mesh:
    '''
    Combines any number of scatterers into a single scatterer\n
    :param scatterers: any number of scatterers to combine
    :param flag: Value will be passed to `vedo.merge`
    :return: the combined scatterer
    '''
    names = []
    Fnames = []
    for scatterer in scatterers:
        names.append(scatterer_file_name(scatterer))
        Fnames.append(scatterer.metadata["FILE"][0])
    
    if flag:
        combined = vedo.merge(scatterers, flag=True)
    else:
        combined = vedo.merge(scatterers)
    combined.filename = "".join(names)
    combined.metadata["FILE"] = "".join(Fnames)
    return combined


def calculate_features(scatterer:Mesh, compute_areas:bool = True, compute_normals:bool=True):
    '''
    @private
    '''
    if compute_areas: scatterer.compute_cell_size()
    if compute_normals: scatterer.compute_normals()

    scatterer.filename = scatterer_file_name(scatterer)
    scatterer.metadata["FILE"] = scatterer.filename.split(".")[0]


