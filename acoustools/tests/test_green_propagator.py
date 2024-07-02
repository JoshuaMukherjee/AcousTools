from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD
from acoustools.Solvers import wgs

from acoustools.Visualiser import Visualise

import torch

board = BOTTOM_BOARD
p = create_points(1,1, y=0, max_pos=0.01, min_pos=-0.01)
print(p)

green = green_propagator(p, board)

x = wgs(p, board=board)
# x = add_lev_sig(x)


A = torch.tensor((-0.06,0, 0.06))
B = torch.tensor((0.06,0, 0.06))
C = torch.tensor((-0.06,0, -0.06))

print(propagate_abs(x,p,A=green))

# Visualise(A,B,C, x, points=p,colour_functions=[propagate_abs,propagate_abs], colour_function_args=[{"A_function":green_propagator},{}], res=(200,200))
Visualise(A,B,C, x, points=p,colour_functions=[propagate_abs], colour_function_args=[{"A_function":green_propagator, 'board':board}], res=(400,400))