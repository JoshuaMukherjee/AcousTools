from acoustools.Utilities import device, DTYPE
import acoustools.Constants as Constants

import vedo, torch
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from vedo import Mesh


def board_name(board:Tensor) -> str:
    '''
    Returns the name for a board, TOP and/or BOTTOM, used in cache system
    :param board: The board to use
    :return: name of board as `<'TOP'><'BOTTOM'><M>` for `M` transducers in the boards 
    '''
    M = board.shape[0]

    top = "TOP" if 1 in torch.sign(board[:,2]) else ""
    bottom = "BOTTOM" if -1 in torch.sign(board[:,2]) else ""
    return top+bottom+str(M)

def scatterer_file_name(scatterer:Mesh) ->str:
    '''
    Get a unique name to describe a scatterer position, calls `str(scatterer.coordinates)`
    ONLY USE TO SET FILENAME, USE `scatterer.filename` TO GET
    :param scatterer: The Mesh to use
    :return: Scatterer name
    
    '''

    f_name = str(list(scatterer.coordinates))
    return f_name

def load_scatterer(path:str, compute_areas:bool = True, compute_normals:bool=True, dx:float=0,dy:float=0,dz:float=0, rotx:float=0, roty:float=0, rotz:float=0, root_path:str="") -> Mesh:
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
    scatterer = vedo.load(root_path+path)
    
    if scatterer is not None:
        if compute_areas: scatterer.compute_cell_size()
        if compute_normals: 
            scatterer.compute_normals()

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


    return scatterer

def calculate_features(scatterer:Mesh, compute_areas:bool = True, compute_normals:bool=True):
    '''
    @private
    '''
    if compute_areas: scatterer.compute_cell_size()
    if compute_normals: scatterer.compute_normals()

    scatterer.filename = scatterer_file_name(scatterer)
    scatterer.metadata["FILE"] = scatterer.filename.split(".")[0]


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


def scale_to_diameter(scatterer:Mesh , diameter: float) -> None:
    '''
    Scale a mesh to a given diameter in the x-axis and recomputes normals and areas \n
    Modifies scatterer in place so does not return anything.\n

    :param scatterer: The scatterer to scale
    :param diameter: The diameter target
    '''
    x1,x2,y1,y2,z1,z2 = scatterer.bounds()
    diameter_sphere = x2 - x1
    scatterer.scale(diameter/diameter_sphere,reset=True)
    scatterer.compute_cell_size()
    scatterer.compute_normals()
    scatterer.filename = scatterer_file_name(scatterer)
    

def get_plane(scatterer: Mesh, origin:tuple[int]=(0,0,0), normal:tuple[int]=(1,0,0)) -> Mesh:
    '''
    Get intersection of a scatterer and a plane\n
    :param scatterer: The scatterer to intersect
    :param origin: A point on the plane as a tuple `(x,y,z)`. Default `(0,0,0)`
    :param normal: The normal to the plane at `point` as a tuple (x,y,z). Default `(1,0,0)`
    :return: new `Mesh` Containing the intersection of the plane and the scatterer
    '''
    intersection = scatterer.clone().intersect_with_plane(origin,normal)
    intersection.filename = scatterer.filename + "plane" + str(origin)+str(normal)
    return intersection

def get_lines_from_plane(scatterer:Mesh, origin:tuple[int]=(0,0,0), normal:tuple[int]=(1,0,0)) -> list[int]:
    '''
    Gets the edges on a plane from the intersection between a scatterer and the plane\n
    :param scatterer: The scatterer to intersect
    :param origin: A point on the plane as a tuple `(x,y,z)`. Default `(0,0,0)`
    :param normal: The normal to the plane at `point` as a tuple (x,y,z). Default `(1,0,0)`
    :return: a list of edges in the plane 
    '''

    mask = [0,0,0]
    for i in range(3):
        mask[i] =not normal[i]
    mask = np.array(mask)

    intersection = get_plane(scatterer, origin, normal)
    verticies = intersection.vertices
    lines = intersection.lines

    connections = []

    for i in range(len(lines)):
        connections.append([verticies[lines[i][0]][mask],verticies[lines[i][1]][mask]])

    return connections

def plot_plane(connections:list[int]) -> None:
    '''
    Plot a set of edges assuming they are co-planar\n
    :param connections: list of connections to plot
    '''
    
    for con in connections:
        xs = [con[0][0], con[1][0]]
        ys = [con[0][1], con[1][1]]
        plt.plot(xs,ys,color = "blue")

    plt.xlim((-0.06,0.06))
    plt.ylim((-0.06,0.06))
    plt.show()

def get_normals_as_points(*scatterers:Mesh, permute_to_points:bool=True) -> Tensor:
    '''
    Returns the normal vectors to the surface of a scatterer as a `torch` `Tensor` as acoustools points\n
    :param scatterers: The scatterer to use
    :param permute_to_points: If true will permute the order of coordinates to agree with what acoustools expects.
    :return: normals
    '''
    norm_list = []
    for scatterer in scatterers:
        scatterer.compute_normals()
        norm =  torch.tensor(scatterer.cell_normals).to(device)

        if permute_to_points:
            norm = torch.permute(norm,(1,0))
        
        norm_list.append(norm.to(DTYPE))
    
    return torch.stack(norm_list)

def get_centre_of_mass_as_points(*scatterers:Mesh, permute_to_points:bool=True) ->Tensor:
    '''
    Returns the centre of mass(es) of a scatterer(s) as a `torch` `Tensor` as acoustools points\n
    :param scatterers: The scatterer(s) to use
    :param permute_to_points: If true will permute the order of coordinates to agree with what acoustools expects.
    :return: centre of mass(es)
    '''
    centres_list = []
    for scatterer in scatterers:
        centre_of_mass =  torch.tensor(scatterer.center_of_mass()).to(device)

        if permute_to_points:
            centre_of_mass = torch.unsqueeze(centre_of_mass,1)
        
        centres_list.append(centre_of_mass.to(DTYPE))
    
    return torch.real(torch.stack(centres_list))


def get_centres_as_points(*scatterers:Mesh, permute_to_points:bool=True, add_normals:bool=False, normal_scale:float=0.001) ->Tensor:
    '''
    Returns the centre of scatterer faces as a `torch` `Tensor` as acoustools points\n
    :param scatterers: The scatterer to use
    :param permute_to_points: If `True` will permute the order of coordinates to agree with what acoustools expects.
    :return: centres
    '''
    centre_list = []
    for scatterer in scatterers:
        centres =  torch.tensor(scatterer.cell_centers).to(device)

        if permute_to_points:
            centres = torch.permute(centres,(1,0)).unsqueeze_(0)
        
        if add_normals:
            norms= get_normals_as_points(scatterer)
            centres += norms.real * normal_scale
        
        centre_list.append(centres.to(DTYPE))
        centres = torch.cat(centre_list,dim=0)
    return centres

def get_areas(*scatterers: Mesh) -> Tensor:
    '''
    Returns the areas of faces of any number of scatterers\n
    :param scatterers: The scatterers to use.
    :return: areas
    '''
    area_list = []
    for scatterer in scatterers:
        scatterer.compute_cell_size()
        area_list.append(torch.Tensor(scatterer.celldata["Area"]).to(device))
    
    return torch.stack(area_list)

def get_weight(scatterer:Mesh, density:float=Constants.p_p, g:float=9.81) -> float:
    '''
    Get the weight of a scatterer\\
    :param scatterer: The scatterer to use\\
    :param density: The density to use. Default density for EPS\\
    :param g: value for g to use. Default 9.81\\
    :return: weight
    '''
    mass = scatterer.volume() * density
    return g * mass

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


def get_edge_data(scatterer:Mesh, wavelength:float=Constants.wavelength, print_output:bool=True, break_down_average:bool=False) -> None|tuple[float]:
    '''
    Get the maximum, minimum and average size of edges in a mesh. Optionally prints or returns the result.\n
    :param scatterer: Mesh of interest
    :param wavelength: Wavenelgth size for printing results as multiple of some wavelength
    :param print_output: If True, prints results else returns values
    :break_down_average: If True will also return (distance_sum, N)
    :return: None if `print_outputs` is `True` else returns `(max_distance, min_distance, average_distance)` and optionally  (distance_sum, N)

    '''
    points = scatterer.vertices

    distance_sum = 0
    N = 0

    max_distance = 0
    min_distance = 100000000


    for (start,end) in scatterer.edges:
        start_point = points[start]
        end_point = points[end]
        sqvec = torch.Tensor((start_point-end_point)**2)
        # print(sqvec, torch.sum(sqvec)**0.5)
        distance = torch.sum(sqvec)**0.5
        distance_sum += distance
        N += 1
        if distance < min_distance:
            min_distance = distance
        if distance > max_distance:
            max_distance = distance

    average_distance = distance_sum/N

    if print_output:
        print('Max Distance', max_distance.item(),'=' ,max_distance.item()/wavelength, 'lambda')
        print('Min Distance', min_distance.item(),'=', min_distance.item()/wavelength, 'lambda')
        print('Ave Distance', average_distance.item(),'=', average_distance.item()/wavelength, 'lambda')
    else:
        if break_down_average:
            return (max_distance, min_distance, average_distance), (distance_sum, N)
        else:
            return (max_distance, min_distance, average_distance)