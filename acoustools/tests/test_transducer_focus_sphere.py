from acoustools.Utilities import forward_model, create_points, propagate_abs, BOARD_POSITIONS
from acoustools.Visualiser import Visualise, ABC


import math,torch

import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt


def sphere(n, r):
    points = []
    for i in range(n):
        x = r * math.sin(i) * math.cos(n*i)
        y = r * math.sin(i) * math.sin(n*i)
        z = r * math.cos(i)
        points.append([x,y,z])

    points = torch.Tensor(points)
    return points




board = sphere(512, BOARD_POSITIONS)

p=create_points(1,1,0,0,0)
F = forward_model(p,board)
x = torch.exp(1j*torch.zeros((512,1)))


A,B,C = ABC(0.13)

Visualise(A,B,C,x, colour_functions=[propagate_abs], colour_function_args=[{'board':board}], res=(300,300))


# print(points)

# plt.figure().add_subplot(111, projection='3d').scatter(points[:,0],points[:,1],points[:,2] )
# plt.show()
