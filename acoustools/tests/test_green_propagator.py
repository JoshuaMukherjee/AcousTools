from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD
from acoustools.Solvers import wgs

from acoustools.Visualiser import Visualise, Visualise_single

import torch

import matplotlib.pyplot as plt

board = BOTTOM_BOARD
p = create_points(3,1, y=0)
print(p)

green = green_propagator(p, board)

print(green)
print(green.mH)
print(green @ green.mH)

x = wgs(p, board=board, A=green)
# x = add_lev_sig(x)


A = torch.tensor((-0.06,0, 0.06))
B = torch.tensor((0.06,0, 0.06))
C = torch.tensor((-0.06,0, -0.06))

print(propagate_abs(x,p,A=green))

green_im =30*Visualise_single(A,B,C,x,propagate_abs,{"A_function":green_propagator, 'board':board})
pm_im = Visualise_single(A,B,C,x,propagate_abs,{'board':board})

# plt.matshow((pm_im/green_im).cpu().detach())
# plt.colorbar()
# plt.show()


Visualise(A,B,C, x, points=p,colour_functions=[propagate_abs,propagate_abs], colour_function_args=[{"A_function":green_propagator, 'board':board},{'board':board}], res=(200,200))
# Visualise(A,B,C, x, points=p,colour_functions=[propagate_abs], colour_function_args=[{"A_function":green_propagator, 'board':board}], res=(400,400))