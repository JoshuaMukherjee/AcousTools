from acoustools.Utilities import device, DTYPE
import acoustools.Constants as Constants

import vedo, torch
import matplotlib.pyplot as plt
import numpy as np


def board_name(board):
    '''
    Returns the name for a board, TOP and/or BOTTOM, used in cache system\\
    `board` The board to use
    Returns name of board as `<'TOP'><'BOTTOM'><M>` for `M` transducers in the boards 
    '''
    M = board.shape[0]

    top = "TOP" if 1 in torch.sign(board[:,2]) else ""
    bottom = "BOTTOM" if -1 in torch.sign(board[:,2]) else ""
    return top+bottom+str(M)

def scatterer_file_name(scatterer):
    '''
    Get a unique name to describe a scatterer position, calls `str(scatterer.coordinates)`\\
    `scatterer` The Mesh to use\\
    Returns the name\\
    ONLY USE TO SET FILENAME, USE `scatterer.filename` TO GET
    '''

    f_name = str(scatterer.coordinates)


    return f_name

def load_scatterer(path, compute_areas = True, compute_normals=True, dx=0,dy=0,dz=0, rotx=0, roty=0, rotz=0, root_path=""):
    '''
    Loads a scatterer as a `vedo` `Mesh` and applies translations as needed\\
    `path` The name of the scatterer to load\\
    `compute_areas` if true will call `scatterer.compute_cell_size()`. Default True\\
    `compute_normals` if true will call `scatterer.compute_normals()`. Default True\\
    `dx` Translation in the x direction to apply\\
    `dy` Translation in the y direction to apply\\
    `dz` Translation in the z direction to apply\\
    `rotx` Rotation around the x axis to apply\\
    `roty` Rotation around the y axis to apply\\
    `rotz` Rotation around the z axis to apply\\
    `root_path` The folder containing the file, the scatterer to be loaded will be loaded from `root_path+path`\\
    Returns the `vedo` `Mesh` of the scatterer
    '''
    scatterer = vedo.load(root_path+path)
    if compute_areas: scatterer.compute_cell_size()
    if compute_normals: 
        scatterer.compute_normals()

    scatterer.metadata["rotX"] = 0
    scatterer.metadata["rotY"] = 0
    scatterer.metadata["rotZ"] = 0

    scatterer.filename = scatterer.filename.split("/")[-1]

    scatterer.metadata["FILE"] = scatterer.filename.split(".")[0]


    rotate(scatterer,(1,0,0),rotx)
    rotate(scatterer,(0,1,0),roty)
    rotate(scatterer,(0,0,1),rotz)

    translate(scatterer,dx,dy,dz)


    return scatterer

def load_multiple_scatterers(paths,  compute_areas = True, compute_normals=True, dxs=[],dys=[],dzs=[], rotxs=[], rotys=[], rotzs=[], root_path=""):
    '''
    Loads multiple scatterers and combines them into a single scatterer object\\
    `path` The name of the scatterer to load\\
    `compute_areas` if true will call `scatterer.compute_cell_size()`. Default True\\
    `compute_normals` if true will call `scatterer.compute_normals()`. Default True\\
    `dxs` List of translations in the x direction to apply to each scatterer\\
    `dys` List of translations in the y direction to apply to each scatterer\\
    `dzs` List of translations in the z direction to apply to each scatterer\\
    `rotxs` List pf rotations around the x axis to apply to each scatterer\\
    `rotys` List pf rotations around the y axis to apply to each scatterer\\
    `rotzs` List pf rotations around the z axis to apply to each scatterer\\
    `root_path` The folder containing the file, the scatterer to be loaded will be loaded from `root_path+path`\\
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

def merge_scatterers(*scatterers, flag=False):
    '''
    Combines any number of scatterers into a single scatterer\\
    `scatterers` any number of scatterers to combine\\
    `flag` Value will be passed to `vedo.merge`\\
    Returns the combined scatterer
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


def scale_to_diameter(scatterer, diameter):
    '''
    Scale a mesh to a given diameter in the x-axis and recomputes normals and areas\\
    `scatterer` The scatterer to scale\\
    `diameter` The diameter target\\
    Modifies scatterer in place so does not return anything.
    '''
    x1,x2,y1,y2,z1,z2 = scatterer.bounds()
    diameter_sphere = x2 - x1
    scatterer.scale(diameter/diameter_sphere,reset=True)
    scatterer.compute_cell_size()
    scatterer.compute_normals()
    scatterer.filename = scatterer_file_name(scatterer)
    

def get_plane(scatterer, origin=(0,0,0), normal=(1,0,0)):
    '''
    Get intersection of a scatterer and a plane\\
    `scatterer` The scatterer to intersect\\
    `origin` A point on the plane as a tuple `(x,y,z)`. Default `(0,0,0)`\\
    `normal` The normal to the plane at `point` as a tuple (x,y,z). Default `(1,0,0)`\\
    Returns new `Mesh` Containing the intersection of the plane and the scatterer
    '''
    intersection = scatterer.clone().intersect_with_plane(origin,normal)
    intersection.filename = scatterer.filename + "plane" + str(origin)+str(normal)
    return intersection

def get_lines_from_plane(scatterer, origin=(0,0,0), normal=(1,0,0)):
    '''
    Gets the edges on a plane from the intersection between a scatterer and the plane\\
    `scatterer` The scatterer to intersect\\
    `origin` A point on the plane as a tuple `(x,y,z)`. Default `(0,0,0)`\\
    `normal` The normal to the plane at `point` as a tuple (x,y,z). Default `(1,0,0)`\\
    Returns a list of edges in the plane 
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

def plot_plane(connections):
    '''
    Plot a set of edges assuming they are co-planar\\
    `connections` list of connections to plot\\
    '''
    
    for con in connections:
        xs = [con[0][0], con[1][0]]
        ys = [con[0][1], con[1][1]]
        plt.plot(xs,ys,color = "blue")

    plt.xlim((-0.06,0.06))
    plt.ylim((-0.06,0.06))
    plt.show()

def get_normals_as_points(*scatterers, permute_to_points=True):
    '''
    Returns the normal vectors to the surface of a scatterer as a `torch` `Tensor` as acoustools points\\
    `scatterers` The scatterer to use\\
    `permute_to_points` If true will permute the order of coordinates to agree with what acoustools expects.\\
    returns normals
    '''
    norm_list = []
    for scatterer in scatterers:
        scatterer.compute_normals()
        norm =  torch.tensor(scatterer.cell_normals).to(device)

        if permute_to_points:
            norm = torch.permute(norm,(1,0))
        
        norm_list.append(norm.to(DTYPE))
    
    return torch.stack(norm_list)

def get_centre_of_mass_as_points(*scatterers, permute_to_points=True):
    '''
    Returns the centre of mass(es) of a scatterer(s) as a `torch` `Tensor` as acoustools points\\
    `scatterers` The scatterer(s) to use\\
    `permute_to_points` If true will permute the order of coordinates to agree with what acoustools expects.\\
    returns centre of mass(es)
    '''
    centres_list = []
    for scatterer in scatterers:
        centre_of_mass =  torch.tensor(scatterer.center_of_mass()).to(device)

        if permute_to_points:
            centre_of_mass = torch.unsqueeze(centre_of_mass,1)
        
        centres_list.append(centre_of_mass.to(DTYPE))
    
    return torch.real(torch.stack(centres_list))


def get_centres_as_points(*scatterers, permute_to_points=True, add_normals=False, normal_scale=0.001):
    '''
    Returns the centre of scatterer faces as a `torch` `Tensor` as acoustools points\\
    `scatterers` The scatterer to use\\
    `permute_to_points` If true will permute the order of coordinates to agree with what acoustools expects.\\
    returns centres
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

def get_areas(*scatterers):
    '''
    Returns the areas of faces of any number of scatterers\\
    `scatterers` The scatterers to use.
    Returns areas
    '''
    area_list = []
    for scatterer in scatterers:
        scatterer.compute_cell_size()
        area_list.append(torch.Tensor(scatterer.celldata["Area"]).to(device))
    
    return torch.stack(area_list)

def get_weight(scatterer, density=Constants.p_p, g=9.81):
    '''
    Get the weight of a scatterer\\
    `scatterer` The scatterer to use\\
    `density` The density to use. Default density for EPS\\
    `g` value for g to use. Default 9.81\\
    Returns weight
    '''
    mass = scatterer.volume() * density
    return g * mass

def translate(scatterer, dx=0,dy=0,dz=0):
    '''
    Translates a scatterer by (dx,dy,dz)\\
    `scatterer` The scatterer to use\\
    `dx` Translation in the x direction\\
    `dy` Translation in the y direction\\
    `dz` Translation in the z direction\\
    Modifies inplace so does not return a value
    '''
    scatterer.shift(np.array([dx,dy,dz]))
    scatterer.filename = scatterer_file_name(scatterer)

def rotate(scatterer, axis, rot, centre=()):
    '''
    Rotates a scatterer in axis by rot\\
    `scatterer` The scatterer to use\\
    `axis` The axis to rotate in\\
    `rot` Angle to rotate\\
    Modifies inplace so does not return a value
    '''
    if axis[0]:
        scatterer.metadata["rotX"] = scatterer.metadata["rotX"] + rot
    if axis[1]:
        scatterer.metadata["rotY"] = scatterer.metadata["rotY"] + rot
    if axis[2]:
        scatterer.metadata["rotZ"] = scatterer.metadata["rotZ"] + rot
    scatterer.rotate(rot, axis,center=centre)
    scatterer.filename = scatterer_file_name(scatterer)


def downsample(scatterer, factor=2, n=None, method='quadric', boundaries=False, compute_areas=True, compute_normals=True):
    '''
    Downsamples a mesh to have `factor` less elements\\
    `scatterer` The scatterer to use\\
    `factor` The factor to downsample\\
    `method`, `boundaries` - passed to `vedo.decimate`\\
    `compute_areas` if true will call `scatterer.compute_cell_size()`. Default True\\
    `compute_normals` if true will call `scatterer.compute_normals()`. Default True\\
    Returns downsampled mesh
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


