import torch
from acoustools.BEM.Force import BEM_compute_force
from acoustools.Utilities import create_points, TRANSDUCERS
from acoustools.Constants import V

from torch import Tensor
from vedo import Mesh

def stiffness_finite_differences_BEM(activations:Tensor, points:Tensor, board:Tensor|None=None, scatterer:Mesh = None, path=None, H=None, V=V, delta= 0.001):
    '''
    Computes the stiffness at a point as the gradient of the force. Force computed analytically and then finite differences used to find the gradient \n
    Computed as `-1* (Fx + Fy + Fz)` where `Fa` is the gradient of force in that direction \n 
    :param activation: Hologram
    :param points: Points of interest
    :param board: Transducers to use
    :param delta: finite differences step size
    
    '''

    if board is None:
        board = TRANSDUCERS

    dx = create_points(1,1,delta,0,0)
    dy = create_points(1,1,0,delta,0)
    dz = create_points(1,1,0,0,delta)

    Fx1 = BEM_compute_force(activations,points + dx,board=board,scatterer=scatterer, path=path, H=H, V=V)[0]
    Fx2 = BEM_compute_force(activations,points - dx,board=board,scatterer=scatterer, path=path, H=H, V=V)[0]

    Fx = ((Fx1 - Fx2) / (2*delta))

    Fy1 = BEM_compute_force(activations,points + dy,board=board,scatterer=scatterer, path=path, H=H, V=V)[1]
    Fy2 = BEM_compute_force(activations,points - dy,board=board,scatterer=scatterer, path=path, H=H, V=V)[1]

    Fy = ((Fy1 - Fy2) / (2*delta))

    Fz1 = BEM_compute_force(activations,points + dz,board=board,scatterer=scatterer, path=path, H=H, V=V)[2]
    Fz2 = BEM_compute_force(activations,points - dz,board=board,scatterer=scatterer, path=path, H=H, V=V)[2]
    
    Fz = ((Fz1 - Fz2) / (2*delta))

    return -1* (Fx + Fy + Fz)
