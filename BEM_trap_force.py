from acoustools.Utilities import *
from acoustools.BEM import BEM_gorkov_analytical
import torch
from torch import Tensor
from vedo import Mesh

def BEM_trap_force(activations: Tensor, points: Tensor, scatterer: Mesh|None|str=None,
                       board: Tensor|None=None, H: Tensor|None=None, E: Tensor|None=None, 
                       delta: float=1e-6, dims: str='XYZ', **params) -> Tensor:
    '''
    Calculates acoustic radiation trap force using finite differences of the Gorkov potential (taking negative gradient)
    
    :param activations: Transducer hologram
    :param points: Points to calculate force at
    :param scatterer: The mesh used (as a `vedo` `mesh` object) or string of path to mesh
    :param board: Transducers to use 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param delta: Small displacement for finite difference calculation
    :param dims: Dimensions to calculate force in (default: XYZ)
    :return: Force vector for each point [Fx, Fy, Fz]
    '''
    #Check if points is properly batched
    if len(points.shape) < 3:
        #Add batch dimension if needed
        points = points.unsqueeze(0)
    
    #Points tensor shape: [B, 3, N] from create_points() -> B x 3 x N
    B = points.shape[0]  #Batch size
    N = points.shape[2]  #Number of points
    
    #Initialise force tensor
    force = torch.zeros(B, 3, N, device=device)
    
    #Calculate X component of force
    if 'X' in dims.upper():
        #Points slightly positive in X
        points_plus_x = points.clone()
        points_plus_x[:, 0, :] += delta
        
        #Points slightly negative in X
        points_minus_x = points.clone()
        points_minus_x[:, 0, :] -= delta
        
        #Calculate Gor'kov potential at offset points
        U_plus_x = BEM_gorkov_analytical(activations, points_plus_x, scatterer, board, H, E, dims=dims, **params)
        U_minus_x = BEM_gorkov_analytical(activations, points_minus_x, scatterer, board, H, E, dims=dims, **params)
        
        #Finite difference approximation of gradient (central difference)
        #2*delta is is x difference between points_plus_x and points_minus_x
        dU_dx = (U_plus_x - U_minus_x) / (2 * delta)
        
        #Force is negative gradient
        force[:, 0, :] = -dU_dx
    
    #Calculate Y component of force
    if 'Y' in dims.upper():
        points_plus_y = points.clone()
        points_plus_y[:, 1, :] += delta
        
        points_minus_y = points.clone()
        points_minus_y[:, 1, :] -= delta
        
        U_plus_y = BEM_gorkov_analytical(activations, points_plus_y, scatterer, board, H, E, dims=dims, **params)
        U_minus_y = BEM_gorkov_analytical(activations, points_minus_y, scatterer, board, H, E, dims=dims, **params)
        
        dU_dy = (U_plus_y - U_minus_y) / (2 * delta)
    
        force[:, 1, :] = -dU_dy
    
    #Calculate Z component of force
    if 'Z' in dims.upper():
        points_plus_z = points.clone()
        points_plus_z[:, 2, :] += delta
        
        points_minus_z = points.clone()
        points_minus_z[:, 2, :] -= delta
        
        U_plus_z = BEM_gorkov_analytical(activations, points_plus_z, scatterer, board, H, E, dims=dims, **params)
        U_minus_z = BEM_gorkov_analytical(activations, points_minus_z, scatterer, board, H, E, dims=dims, **params)
        
        dU_dz = (U_plus_z - U_minus_z) / (2 * delta)
        
        force[:, 2, :] = -dU_dz
    
    return force