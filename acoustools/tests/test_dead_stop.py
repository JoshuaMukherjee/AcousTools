from acoustools.Utilities import create_points, TRANSDUCERS, propagate_abs
from acoustools.Paths import interpolate_points, distance
from acoustools.Solvers import wgs, gradient_descent_solver
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force
from acoustools.Visualiser import Visualise, ABC

from torch import Tensor
import torch

path = []

board = TRANSDUCERS

start = create_points(1,1,-0.02,-0.015,0)
end = create_points(1,1,0.02,0.015,0)

N = 500 #5cm in 500 steps = 0.1mm

path = interpolate_points(start,end,N)

xs = []

# for i,p in enumerate(path):
#     print(i, end = '\r')
    
#     x = wgs(p, board=board)

#     xs.append(x)

def grad_dead_stop(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params):
    Uz = gorkov_analytical(transducer_phases, points, board, axis='Z')
    direction_norm  = ((end - start) / distance(start, end)).squeeze()
    F = compute_force(transducer_phases, points, board).squeeze()

    F_dir = torch.dot(F, direction_norm) * direction_norm

    objective = Uz - 10*torch.sum(F_dir) #Maximise Force while minimising Gorkov in Z
    objective = objective.reshape((1,))
    # print(Uz, F_dir, objective)

    return objective


iters = 1000
lr =0.1

x_stop = gradient_descent_solver(end, grad_dead_stop, board, log=False, iters=iters, lr=lr)

abc = ABC(0.02, plane = 'xy', origin=end)

Visualise(*abc, activation=x_stop, points=end, colour_functions=[propagate_abs, gorkov_analytical], link_ax='none')