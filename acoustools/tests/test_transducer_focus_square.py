from acoustools.Utilities import forward_model, create_points, propagate_abs, TRANSDUCERS
from acoustools.Visualiser import Visualise, ABC
from acoustools.Constants import k


import torch


board = TRANSDUCERS

# p=create_points(1,1,0,0,0)
p = create_points(1,1, y=0)
print(p)
F = forward_model(p,board)


distance = torch.sqrt(torch.sum((board.unsqueeze(0).mT - p.expand(1,3,512))**2,axis=1))


x = torch.exp(1j*(torch.zeros((512,1)) - distance.T*k))


A,B,C = ABC(0.13)

Visualise(A,B,C,x, points=p,colour_functions=[propagate_abs], colour_function_args=[{'board':board}], res=(300,300))

