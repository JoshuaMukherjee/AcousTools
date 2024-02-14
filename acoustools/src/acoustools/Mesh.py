from acoustools.Utilities import device, forward_model_batched, DTYPE
import acoustools.Constants as Constants

import vedo, torch
import matplotlib.pyplot as plt
import numpy as np


def board_name(board):
    M = board.shape[0]

    top = "TOP" if 1 in torch.sign(board[:,2]) else ""
    bottom = "BOTTOM" if -1 in torch.sign(board[:,2]) else ""
    return top+bottom+str(M)

def scatterer_file_name(scatterer):
    '''
    ONLY USE TO SET FILENAME, USE `scatterer.filename` TO GET
    '''
    # print(torch.sign(board[:,2]))
    # print(top,bottom)
    f_name = scatterer.metadata["FILE"][0]
    bounds = [str(round(i,8)) for i in scatterer.bounds()]
    # centre_of_mass = get_centre_of_mass_as_points(scatterer)
    # centre_of_mass = str(centre_of_mass[0,0].item()) +  "-" + str(centre_of_mass[0,1].item()) + "-" + str(centre_of_mass[0,2].item())
    rots = str(scatterer.metadata["rotX"][0]) + str(scatterer.metadata["rotY"][0]) + str(scatterer.metadata["rotZ"][0])
    # if "\\" in f_name:
        # f_name = f_name.split("/")[1].split(".")[0]
    f_name = f_name + "".join(bounds) +"--" + "-".join(rots)
    return f_name

def load_scatterer(path, compute_areas = True, compute_normals=True, dx=0,dy=0,dz=0, rotx=0, roty=0, rotz=0, root_path=""):
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
    x1,x2,y1,y2,z1,z2 = scatterer.bounds()
    diameter_sphere = x2 - x1
    scatterer.scale(diameter/diameter_sphere,reset=True)
    scatterer.filename = scatterer_file_name(scatterer)
    

def get_plane(scatterer, origin=(0,0,0), normal=(1,0,0)):
    intersection = scatterer.clone().intersect_with_plane(origin,normal)
    intersection.filename = scatterer.filename + "plane" + str(origin)+str(normal)
    return intersection

def get_lines_from_plane(scatterer, origin=(0,0,0), normal=(1,0,0)):

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
    
    for con in connections:
        xs = [con[0][0], con[1][0]]
        ys = [con[0][1], con[1][1]]
        plt.plot(xs,ys,color = "blue")

    plt.xlim((-0.06,0.06))
    plt.ylim((-0.06,0.06))
    plt.show()

def get_normals_as_points(*scatterers, permute_to_points=True):
    norm_list = []
    for scatterer in scatterers:
        scatterer.compute_normals()
        norm =  torch.tensor(scatterer.cell_normals).to(device)

        if permute_to_points:
            norm = torch.permute(norm,(1,0))
        
        norm_list.append(norm.to(DTYPE))
    
    return torch.stack(norm_list)

def get_centre_of_mass_as_points(*scatterers, permute_to_points=True):
    centres_list = []
    for scatterer in scatterers:
        centre_of_mass =  torch.tensor(scatterer.center_of_mass()).to(device)

        if permute_to_points:
            centre_of_mass = torch.unsqueeze(centre_of_mass,1)
        
        centres_list.append(centre_of_mass.to(DTYPE))
    
    return torch.real(torch.stack(centres_list))


def get_centres_as_points(*scatterers, permute_to_points=True, add_normals=False, normal_scale=0.001):
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
    area_list = []
    for scatterer in scatterers:
        scatterer.compute_cell_size()
        area_list.append(torch.Tensor(scatterer.celldata["Area"]).to(device))
    
    return torch.stack(area_list)

def get_weight(scatterer, density=Constants.p_p, g=9.81):
    mass = scatterer.volume() * density
    return g * mass

def translate(scatterer, dx=0,dy=0,dz=0):
    scatterer.shift(np.array([dx,dy,dz]))
    scatterer.filename = scatterer_file_name(scatterer)

def rotate(scatterer, axis, rot):
    if axis[0]:
        scatterer.metadata["rotX"] = scatterer.metadata["rotX"] + rot
    if axis[1]:
        scatterer.metadata["rotY"] = scatterer.metadata["rotZ"] + rot
    if axis[2]:
        scatterer.metadata["rotZ"] = scatterer.metadata["rotZ"] + rot
    scatterer.rotate(rot, axis)
    scatterer.filename = scatterer_file_name(scatterer)


def downsample(scatterer, factor=2, n=None, method='quadric', boundaries=False, compute_areas=True, compute_normals=True):
    scatterer_small =  scatterer.decimate(1/factor, n, method, boundaries)
    
    scatterer_small.metadata["rotX"] = scatterer.metadata["rotX"]
    scatterer_small.metadata["rotY"] = scatterer.metadata["rotY"]
    scatterer_small.metadata["rotZ"] = scatterer.metadata["rotZ"]

    if compute_areas: scatterer_small.compute_cell_size()
    if compute_normals: 
        scatterer_small.compute_normals()

    scatterer_small.filename = scatterer_file_name(scatterer_small)  + "-scale-" + str(factor)


    return scatterer_small


