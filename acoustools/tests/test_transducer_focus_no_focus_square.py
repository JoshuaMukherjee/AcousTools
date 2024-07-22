from acoustools.Utilities import forward_model, create_points, propagate_abs, TRANSDUCERS
from acoustools.Visualiser import Visualise, ABC


import torch



board = TRANSDUCERS

p=create_points(1,1,0,0,0)
F = forward_model(p,board)
x = torch.exp(1j*torch.zeros((512,1)))


A,B,C = ABC(0.13)

Visualise(A,B,C,x, colour_functions=[propagate_abs], colour_function_args=[{'board':board}], res=(300,300))

