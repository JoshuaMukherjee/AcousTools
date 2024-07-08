from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD, create_board, propagate, device, propagate_phase
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength
from acoustools.Visualiser import Visualise

import torch

import matplotlib.animation as animation
import matplotlib.pyplot as plt


z = wavelength
t1 = create_board(2,z=-z)

board = t1

print(board)

x = torch.ones((1,1)) * torch.e**(1j*0)
# x[1,:] = torch.ones((1,1)) * torch.e**(1j*torch.pi)


dz = 0.001
A = torch.tensor((-z,0, z-dz))
B = torch.tensor((z,0, z-dz))
C = torch.tensor((-z,0, -z+dz))




def propagate_abs_norm(activations, points,board=TRANSDUCERS, A=None, A_function=None, A_function_args={}):
    '''
    Propagates a hologram to target points and returns pressure - Same as `torch.abs(propagate(activations, points,board, A))`\\
    `activations` Hologram to use\\
    `points` Points to propagate to\\
    `board` The Transducer array, default two 16x16 arrays\\
    `A` The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`\\
    Returns point pressure
    '''
    if A_function is not None:
        A = A_function(points, board, **A_function_args)

    out = propagate(activations, points,board,A=A)
    return torch.abs(out) / torch.max(torch.abs(out))

def propagate_signed_norm(activations, points,board=TRANSDUCERS, A=None, A_function=None, A_function_args={}):
    '''
    Propagates a hologram to target points and returns pressure - Same as `torch.abs(propagate(activations, points,board, A))`\\
    `activations` Hologram to use\\
    `points` Points to propagate to\\
    `board` The Transducer array, default two 16x16 arrays\\
    `A` The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`\\
    Returns point pressure
    '''
    if A_function is not None:
        A = A_function(points, board, **A_function_args)

    out = propagate(activations, points,board,A=A)
    norm =  torch.abs(out) / torch.max(torch.abs(out))
    phase = propagate_phase(activations, points,board,A=A)

    return norm * torch.sign(phase)

# Visualise(A,B,C, x, points=p,colour_functions=[propagatae_abs,propagate_abs], colour_function_args=[{"A_function":green_propagator},{}], res=(200,200))

im = Visualise(A,B,C, x, colour_functions=[propagate_abs, propagate_phase], colour_function_args=[{'board':board},{'board':board}], res=(200,200), cmaps=['hot','hsv'])
