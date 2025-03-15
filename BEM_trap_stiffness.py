from acoustools.Utilities import *
from acoustools.BEM import BEM_gorkov_analytical
from BEM_trap_force import BEM_trap_force   
import torch
from torch import Tensor
from vedo import Mesh

#use compute stiffness code but replace compute_force with my code

def BEM_trap_stiffness(activations: Tensor, points: Tensor, scatterer: Mesh|None|str=None,
                       board: Tensor|None=None, H: Tensor|None=None, E: Tensor|None=None, 
                       delta: float=0.001, dims: str='XYZ', **params)  -> Tensor:
    '''
    Computes the stiffness at a point as the gradient of the force. Force computed analytically and then finite differences used to find the gradient \n
    Computed as `-1* (Fx + Fy + Fz)` where `Fa` is the gradient of force in that direction \n 
    :param activation: Hologram
    :param points: Points of interest
    :param board: Transducers to use
    :param delta: finite differences step size
    :params: includes any additional parameters such as BEMMedia folder path
    
    '''

    if board is None:
        board = TRANSDUCERS

    dx = create_points(1,1,delta,0,0)
    dy = create_points(1,1,0,delta,0)
    dz = create_points(1,1,0,0,delta)


    #output of BEM_trap_force is trap force (Fx, Fy, Fz):  tensor([[[ 2.3078e-05],
        #  [-3.6380e-06],
        #  [-1.4968e-03]]]
        # so i want to access this
    Fx1 =     
    
    Fx1 = BEM_trap_force(activations,points + dx, scatterer=scatterer,board=board, return_components=True, **params)[0]
    Fx2 = BEM_trap_force(activations,points - dx, scatterer=scatterer, board=board, return_components=True, **params)[0]

    Fx = ((Fx1 - Fx2) / (2*delta))

    Fy1 = BEM_trap_force(activations,points + dy, scatterer=scatterer, board=board, return_components=True, **params)[1]
    Fy2 = BEM_trap_force(activations,points - dy, scatterer=scatterer, board=board, return_components=True, **params)[1]

    Fy = ((Fy1 - Fy2) / (2*delta))

    Fz1 = BEM_trap_force(activations,points + dz, scatterer=scatterer, board=board, return_components=True, **params)[2]
    Fz2 = BEM_trap_force(activations,points - dz, scatterer=scatterer, board=board, return_components=True, **params)[2]
    
    Fz = ((Fz1 - Fz2) / (2*delta))

    return -1* (Fx + Fy + Fz)