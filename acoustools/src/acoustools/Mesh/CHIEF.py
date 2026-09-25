

from acoustools.Utilities import device, DTYPE
import acoustools.Constants as Constants

import torch
from torch import Tensor
from vedo import Mesh

from typing import Literal

from acoustools.Mesh.Features import get_normals_as_points, get_centre_of_mass_as_points, get_diameter
from acoustools.Mesh.Load import load_scatterer, merge_scatterers
from acoustools.Mesh.Transform import centre_scatterer, translate, scale_to_diameter

def get_CHIEF_points(scatterer:Mesh, P=30, method:Literal['random', 'uniform', 'volume-random', 'tetra-random']='random', start:Literal['surface', 'centre']='surface', scale=0.001, scale_mode:Literal['abs','diameter-scale']='abs') -> Mesh:
    '''
    Generates internal points that can be used for the CHIEF BEM formulation (or any other reason)\n
    :param scatterer: The scatterer to insert points into
    :param P: Number of points. if P=-1 then P= number of mesh elements
    :param method: The method used to generate points \n
        - random: will move scale metres along each of P randomly selected normals \n
        - uniform:  will move scale metres along each of P uniformly spaced normals (based on order coming from `Mesh.get_normals_as_points`) \n
        - volume-random: will use `vedo.Mesh..generate_random_points` to generate P internal points
    :param start: The point to use as the basis for generating points \n
         - surface: Will step along normals from surface (will step in the -ve normal direction)
         - centre: Will step along normal from centre of mass (will step in +ve normal direction)
    :param scale: The distance in m to step 
    :returns internal points:
    '''

    centre_norms = get_normals_as_points(scatterer, permute_to_points=False)

    if scale_mode.lower() == 'diameter-scale':
        d = get_diameter(scatterer)
        scale = scale * d
    

    if start.lower() == 'centre':
        centres = get_centre_of_mass_as_points(scatterer, permute_to_points=False).unsqueeze(1)
        internal_points = centres + centre_norms * scale      

    else:
        centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
        internal_points = centres - centre_norms * scale

    M = centre_norms.shape[1]
    
    if P == -1: P = M

   


    
    if method.lower() == 'random':
        indices = torch.randperm(M)[:P]
        internal_points = internal_points[:, indices,:]

    elif method.lower()== 'uniform':
        idx = [i for i in range(M) if i%(int(M/P)) == 0]
        internal_points = internal_points[:, idx,:]
    elif method.lower() == 'volume-random':
        internal_points = torch.Tensor(scatterer.generate_random_points(P).points).unsqueeze(0)
    elif method.lower() == 'tetra-random':
        tetra = get_tetra_centroids(scatterer, side=0.01)
        indices = torch.randperm(tetra.shape[2])[:P]
        internal_points = tetra[:,:,indices]
        return internal_points
    elif method.lower() == 'tetra-all':
        tetra = get_tetra_centroids(scatterer)
        return tetra

    internal_points = internal_points.permute(0,2,1)


    return internal_points


def get_tetra_centroids(scatterer:Mesh, side=0.02) -> Tensor:

    tetra = scatterer.tetralize(side=side)
    
    cell_centres = tetra.cell_centers()
    
    
    centres =  torch.tensor(cell_centres.points).to(DTYPE).to(device)
    

    return centres.T.unsqueeze_(0)




def insert_parasite(scatterer:Mesh, parasite_path:str = '/Sphere-lam1.stl', root_path:str="../BEMMedia", parasite_size:float=Constants.wavelength/4, parasite_offset:Tensor=None) -> Mesh:
    '''
    Inserts a parasitic body into an existing scatterer. Used to supress the resonance from BEM \n
    See https://doi.org/10.1109/8.310000 \n
    :param scatterer: The scatterer to insert parasite into
    :param parasite_path: The path to the mesh to load and use as parasite
    :param root_path: The folder to load the file from
    :param parasite_size: The diameter to scale the parasite to
    :param parasite_offset: Tensor of offsets for the parasite from the (0,0,0) point
    :returns: Scatterer with parasite inserted
    '''
    parasite = load_scatterer(parasite_path, root_path=root_path)
    centre_scatterer(parasite)
    if parasite_offset is None:
        parasite_offset = get_centre_of_mass_as_points(scatterer)

    dx = parasite_offset[:,0].item()
    dy = parasite_offset[:,1].item()
    dz = parasite_offset[:,2].item()

    translate(parasite, dx=dx, dy=dy, dz=dz)

    scale_to_diameter(parasite, parasite_size)

    infected_scatterer = merge_scatterers(scatterer, parasite)

    return infected_scatterer
