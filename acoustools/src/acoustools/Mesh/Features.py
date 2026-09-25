from acoustools.Utilities import device, DTYPE
import acoustools.Constants as Constants

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
from vedo import Mesh


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
        centre_of_mass =  torch.tensor(scatterer.center_of_mass()).to(DTYPE).to(device)

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
        centres =  torch.tensor(scatterer.cell_centers().points).to(DTYPE).to(device)

        if permute_to_points:
            centres = torch.permute(centres,(1,0)).unsqueeze_(0)
        
        if add_normals:
            norms= get_normals_as_points(scatterer)
            centres += norms.real * normal_scale
        
        centre_list.append(centres)
    centres = torch.cat(centre_list,dim=0)
    return centres

def get_verticies_as_points(*scatterers:Mesh):
    '''
    Gets the verticies of a mesh as a Tensor of AcousTools (B,3,N) points \n
    :param Mesh: Mesh to use
    :returns verticies: verticies as points
    '''
   
    vert_list = []
    for scatterer in scatterers:
        vert =  torch.tensor(scatterer.vertices).to(DTYPE).to(device)
        vert_list.append(vert)

    verts = torch.cat(vert_list,dim=0).unsqueeze(0).permute(0,2,1)
    return verts

def get_cell_verticies(*scatterers:Mesh):
    '''
    Gets a tensor of (B,3,M,3) - batch x (xyz) x Faces x (vertex) \n
    :param Mesh: Mesh to use
    :returns verticies: verticies
    '''
    verts = get_verticies_as_points(*scatterers)
    vert_list = []
    for scatterer in scatterers:
        cells = torch.tensor(scatterer.cells)
        N = cells.shape[0]
        cell_indexes = cells.flatten()
        cell_verts = torch.index_select(verts, 2, cell_indexes)
        cell_verts=cell_verts.reshape(1,3,N,3)


        vert_list.append(cell_verts)
    verts = torch.cat(vert_list,dim=0)
    return verts



def get_barycentric_points(*scatterers:Mesh, N=7, sum=True):
    '''
    @private
    '''
    

    if N != 7: raise ValueError("Only N=7 is supported") #Allow for N as a parameter incase it it implemented in future

    cell_verts = get_cell_verticies(*scatterers)

    DUNAVANT_7 = torch.tensor([
    [1/3, 1/3, 1/3, 0.225],
    [0.0597158717, 0.4701420641, 0.4701420641, 0.1323941527],
    [0.4701420641, 0.0597158717, 0.4701420641, 0.1323941527],
    [0.4701420641, 0.4701420641, 0.0597158717, 0.1323941527],
    [0.7974269853, 0.1012865073, 0.1012865073, 0.1259391805],
    [0.1012865073, 0.7974269853, 0.1012865073, 0.1259391805],
    [0.1012865073, 0.1012865073, 0.7974269853, 0.1259391805],
    ])
    DUNAVANT_7_abg = DUNAVANT_7[:,:3].permute(1,0).unsqueeze(0).unsqueeze(0).unsqueeze(0)


    DUNAVANT_7_W = DUNAVANT_7[:,3]

    cell_verts = cell_verts.unsqueeze(-1)
    barycentric_verts = cell_verts * DUNAVANT_7_abg
    if sum: barycentric_verts = barycentric_verts.sum(dim=3)

    return barycentric_verts, DUNAVANT_7_W

    

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


def get_volume(scatterer:Mesh):
    '''
    Returns the volume of a mesh
    '''
    return scatterer.volume()

def get_diameter(scatterer:Mesh):
    x1,x2,y1,y2,z1,z2 = scatterer.bounds()
    diameter_sphere = torch.norm(torch.Tensor([x2,]) - torch.Tensor([x1,]), p=2)
    return diameter_sphere
