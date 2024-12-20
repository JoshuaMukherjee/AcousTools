import torch
from acoustools.Force import compute_force
from acoustools.Utilities import create_points, TRANSDUCERS

from torch import Tensor

def stiffness_finite_differences(activation:Tensor, points:Tensor, board:Tensor|None=None, delta= 0.001):
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



    Fx1 = compute_force(activation,points + dx,board=board)[0]
    Fx2 = compute_force(activation,points - dx,board=board)[0]

    Fx = ((Fx1 - Fx2) / (2*delta))

    Fy1 = compute_force(activation,points + dy,board=board)[1]
    Fy2 = compute_force(activation,points - dy,board=board)[1]

    Fy = ((Fy1 - Fy2) / (2*delta))

    Fz1 = compute_force(activation,points + dz,board=board)[2]
    Fz2 = compute_force(activation,points - dz,board=board)[2]
    
    Fz = ((Fz1 - Fz2) / (2*delta))

    return -1* (Fx + Fy + Fz)
