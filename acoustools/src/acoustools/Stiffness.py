import torch
from acoustools.Force import compute_force
from acoustools.Utilities import create_points, TRANSDUCERS
import acoustools.Constants as c

from torch import Tensor

def stiffness_finite_differences(activations:Tensor, points:Tensor, board:Tensor|None=None, delta= 0.001, V=c.V, p_ref=c.P_ref, 
                  transducer_radius=c.radius, k=c.k,
                 medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p):
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


    Fx1 = compute_force(activations,points + dx,board=board, V=V, p_ref=p_ref, transducer_radius=transducer_radius, k=k, medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density, particle_speed=particle_speed)[:,:,0]
    Fx2 = compute_force(activations,points - dx,board=board, V=V, p_ref=p_ref, transducer_radius=transducer_radius, k=k, medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density, particle_speed=particle_speed)[:,:,0]

    Fx = ((Fx1 - Fx2) / (2*delta))

    Fy1 = compute_force(activations,points + dy,board=board, V=V, p_ref=p_ref, transducer_radius=transducer_radius, k=k, medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density, particle_speed=particle_speed)[:,:,1]
    Fy2 = compute_force(activations,points - dy,board=board, V=V, p_ref=p_ref, transducer_radius=transducer_radius, k=k, medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density, particle_speed=particle_speed)[:,:,1]

    Fy = ((Fy1 - Fy2) / (2*delta))

    Fz1 = compute_force(activations,points + dz,board=board, V=V, p_ref=p_ref, transducer_radius=transducer_radius, k=k, medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density, particle_speed=particle_speed)[:,:,2]
    Fz2 = compute_force(activations,points - dz,board=board, V=V, p_ref=p_ref, transducer_radius=transducer_radius, k=k, medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density, particle_speed=particle_speed)[:,:,2]
    
    Fz = ((Fz1 - Fz2) / (2*delta))

    return -1* (Fx + Fy + Fz)
