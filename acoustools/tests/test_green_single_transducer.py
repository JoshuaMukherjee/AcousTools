from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD, create_board, propagate
from acoustools.Solvers import wgs

from acoustools.Visualiser import Visualise

import torch

board = create_board(2,z=-0.016)
print(board)

x = torch.ones((1,1)) * torch.e**(1j*0)
# x = add_lev_sig(x)



A = torch.tensor((-0.005,0, -0.005))
B = torch.tensor((0.005,0, -0.005))
C = torch.tensor((-0.005,0, -0.015))


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


# Visualise(A,B,C, x, points=p,colour_functions=[propagate_abs,propagate_abs], colour_function_args=[{"A_function":green_propagator},{}], res=(200,200))
Visualise(A,B,C, x,colour_functions=[propagate_abs_norm, propagate_abs_norm], 
          colour_function_args=[{"A_function":green_propagator, 'board':board}, {'board':board}], res=(200,200), clr_labels=["Normalised Pressure","Normalised Pressure"])