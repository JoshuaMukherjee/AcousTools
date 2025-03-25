
import torch
from torch import Tensor

from vedo import Mesh

import hashlib, pickle

from acoustools.Utilities import DTYPE, forward_model_grad
from acoustools.BEM.Forward_models import get_cache_or_compute_H
from acoustools.Mesh import get_areas, get_centres_as_points, get_normals_as_points
import acoustools.Constants as Constants


 
def get_G_partial(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:
    '''
    Computes gradient of the G matrix in BEM \n
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Ignored
    :param return_components: if true will return the subparts used to compute
    :return: Gradient of the G matrix in BEM
    '''
    #Bk3. Pg. 26
    # if board is None:
    #     board = TRANSDUCERS

    areas = get_areas(scatterer)
    centres = get_centres_as_points(scatterer)
    normals = get_normals_as_points(scatterer)

    N = points.shape[2]
    M = centres.shape[2]


    points = points.unsqueeze(3).expand(-1,-1,-1,M)
    centres = centres.unsqueeze(2).expand(-1,-1,N,-1)

    diff = points - centres    
    diff_square = diff**2
    distances = torch.sqrt(torch.sum(diff_square, 1))
    distances_expanded = distances.unsqueeze(1).expand((1,3,N,M))
    distances_expanded_square = distances_expanded**2

    # G  =  e^(ikd) / 4pi d
    G = torch.exp(1j * Constants.k * distances_expanded) / (4*3.1415*distances_expanded)

    #Ga =  [i*da * e^{ikd} * (kd+i) / 4pi d^2]

    #d = distance
    #da = -(at - a)^2 / d

    da = diff / distances_expanded
    kd = Constants.k * distances_expanded
    phase = torch.exp(1j*kd)
    Ga =  ( (1j*da*phase * (kd + 1j))/ (4*3.1415*distances_expanded_square))

    #P = (ik - 1/d)
    P = (1j*Constants.k - 1/distances_expanded)
    #Pa = da / d^2
    Pa = da / distances_expanded_square

    #C = distance \cdot normals
    C = (diff * normals.unsqueeze(2)).sum(dim=1) / distances

    nx = normals[:,0]
    ny = normals[:,1]
    nz = normals[:,2]

    dx = diff[:,0,:]
    dy = diff[:,1,:]
    dz = diff[:,2,:]

    distances_cubed = distances**3

    Cx = (nx*(dy**2 + dz**2) - dx * (ny*dy + nz*dz)) / distances_cubed
    Cy = (ny*(dx**2 + dz**2) - dy * (nx*dx + nz*dz)) / distances_cubed
    Cz = (nz*(dx**2 + dy**2) - dz * (nx*dx + ny*dy)) / distances_cubed

    Cx.unsqueeze_(1)
    Cy.unsqueeze_(1)
    Cz.unsqueeze_(1)

    Ca = torch.cat([Cx, Cy, Cz],axis=1)

    grad_G = Ga*P*C + G*P*Ca + G*Pa*C

    grad_G = areas * grad_G.to(DTYPE)


    
    return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:]


def BEM_forward_model_grad(points:Tensor, scatterer:Mesh, transducers:Tensor|Mesh=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H:Tensor|None=None, return_components:bool=False,
                           path:str="Media") -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    Computes the gradient of the forward propagation for BEM\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param transducers: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param use_cache_H_grad: If true uses the cache system, otherwise computes `H` and does not save it
    :param print_lines: if true prints messages detaling progress
    :param H: Precomputed `H` - if `None` `H` will be computed
    :param return_components: if true will return the subparts used to compute
    :param path: path to folder containing `BEMCache/` 
    :return: Ex, Ey, Ez
    '''
    if transducers is None:
        transducers = TRANSDUCERS

    B = points.shape[0]
    if H is None:
        H = get_cache_or_compute_H(scatterer,transducers,use_cache_H, path, print_lines)
    
    Fx, Fy, Fz  = forward_model_grad(points, transducers)
    Gx, Gy, Gz = get_G_partial(points, scatterer, transducers)


    Gx[Gx.isnan()] = 0
    Gy[Gy.isnan()] = 0
    Gz[Gz.isnan()] = 0

    Fx = Fx.to(DTYPE)
    Fy = Fy.to(DTYPE)
    Fz = Fz.to(DTYPE)

    H = H.expand(B, -1, -1).to(DTYPE)


    Ex = Fx + Gx@H
    Ey = Fy + Gy@H
    Ez = Fz + Gz@H


    if return_components:
        return Ex.to(DTYPE), Ey.to(DTYPE), Ez.to(DTYPE), Fx, Fy, Fz, Gx, Gy, Gz, H
    else:
        return Ex.to(DTYPE), Ey.to(DTYPE), Ez.to(DTYPE)