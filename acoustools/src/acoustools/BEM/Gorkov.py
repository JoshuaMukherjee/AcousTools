import torch
from torch import Tensor

from vedo import Mesh

from acoustools.Utilities import TRANSDUCERS, device
from acoustools.Mesh import load_scatterer

from acoustools.BEM.Forward_models import compute_E
from acoustools.BEM.Gradients import BEM_forward_model_grad

import acoustools.Constants as Constants

from acoustools.Gorkov  import get_gorkov_constants




def BEM_gorkov_analytical(activations:Tensor,points:Tensor,scatterer:Mesh|None|str=None,
                          board:Tensor|None=None,H:Tensor|None=None,E:Tensor|None=None, return_components:bool=False, dims='XYZ',
                          V:float=Constants.V, p_ref = Constants.P_ref, k=Constants.k, transducer_radius = Constants.radius, 
                        medium_density=Constants.p_0, medium_speed = Constants.c_0, particle_density = Constants.p_p, particle_speed = Constants.c_p,
                          **params) -> Tensor:
    '''
    Returns Gor'kov potential computed analytically from the BEM model\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object) or string of path to mesh
    :param board: Transducers to use 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param return_components: 
    :param dims: Dimensions to consider gradient in
    :param V: Volume of particles
    :return: Gor'kov potential at point U
    '''
    if board is None:
        board = TRANSDUCERS
    if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
    
    path = params['path']
    
    if E is None:
        E = compute_E(scatterer,points,board,H=H,path=path, k=k, p_ref=p_ref, transducer_radius= transducer_radius)

    Ex, Ey, Ez = BEM_forward_model_grad(points,scatterer,board,H=H,path=path)

    p = E@activations
    
    if 'X' in dims.upper():
        px = Ex@activations
    else:
        px = torch.Tensor((0,)).to(device)
    
    if 'Y' in dims.upper():
        py = Ey@activations
    else:
        py = torch.Tensor((0,)).to(device)
    
    if 'Z' in dims.upper():
        pz = Ez@activations
    else:
        pz = torch.Tensor((0,)).to(device)
    
    # K1 = V / (4*Constants.p_0*Constants.c_0**2)
    # K2 = 3*V / (4*(2*Constants.f**2 * Constants.p_0))

    K1, K2 = get_gorkov_constants(V=V, c_0=medium_speed, c_p=particle_speed, p_0=medium_density, p_p=particle_density)

    a = K1 * torch.abs(p)**2 
    b = K2*(torch.abs(px)**2 + torch.abs(py)**2 + torch.abs(pz)**2)

    U = a-b

    if return_components:
        return U, a ,b

    return U
