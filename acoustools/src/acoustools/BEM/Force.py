
from acoustools.BEM import BEM_forward_model_second_derivative_mixed, BEM_forward_model_second_derivative_unmixed, BEM_forward_model_grad, compute_E
from acoustools.Utilities import TRANSDUCERS
import acoustools.Constants as c

from torch import Tensor
import torch

from vedo import Mesh

def BEM_compute_force(activations:Tensor, points:Tensor,board:Tensor|None=None,return_components:bool=False, V=c.V, scatterer:Mesh=None, 
                  H:Tensor=None, path:str="Media") -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Returns the force on a particle using the analytical derivative of the Gor'kov potential and BEM\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param board: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param return_components: If true returns force as one tensor otherwise returns Fx, Fy, Fz
    :param V: Particle volume
    :param scatterer: Scatterer to use
    :param H: H to use, will load/compute if None
    :param path: Path to folder containing BEMCache
    :return: force  
    '''

    #Bk.2 Pg.319

    if board is None:
        board = TRANSDUCERS
    
    F = compute_E(scatterer=scatterer,points=points,board=board,H=H, path=path)
    Fx, Fy, Fz = BEM_forward_model_grad(points,transducers=board,scatterer=scatterer,H=H, path=path)
    Fxx, Fyy, Fzz = BEM_forward_model_second_derivative_unmixed(points,transducers=board,scatterer=scatterer,H=H, path=path)
    Fxy, Fxz, Fyz = BEM_forward_model_second_derivative_mixed(points,transducers=board,scatterer=scatterer,H=H, path=path)

    p   = (F@activations)
    Px  = (Fx@activations)
    Py  = (Fy@activations)
    Pz  = (Fz@activations)
    Pxx = (Fxx@activations)
    Pyy = (Fyy@activations)
    Pzz = (Fzz@activations)
    Pxy = (Fxy@activations)
    Pxz = (Fxz@activations)
    Pyz = (Fyz@activations)


    grad_p = torch.stack([Px,Py,Pz])
    grad_px = torch.stack([Pxx,Pxy,Pxz])
    grad_py = torch.stack([Pxy,Pyy,Pyz])
    grad_pz = torch.stack([Pxz,Pyz,Pzz])


    p_term = p*grad_p.conj() + p.conj()*grad_p

    px_term = Px*grad_px.conj() + Px.conj()*grad_px
    py_term = Py*grad_py.conj() + Py.conj()*grad_py
    pz_term = Pz*grad_pz.conj() + Pz.conj()*grad_pz

    K1 = V / (4*c.p_0*c.c_0**2)
    K2 = 3*V / (4*(2*c.f**2 * c.p_0))

    grad_U = K1 * p_term - K2 * (px_term + py_term + pz_term)
    force = -1*(grad_U).squeeze().real

    if return_components:
        return force[0], force[1], force[2] 
    else:
        return force 

