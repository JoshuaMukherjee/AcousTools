from acoustools.Utilities import create_points, TOP_BOARD, device, DTYPE
from acoustools.Mesh import load_scatterer
from acoustools.BEM import BEM_gorkov_analytical, compute_E, propagate_BEM_pressure

from acoustools.Solvers import gradient_descent_solver

from acoustools.Visualiser import Visualise, ABC

import torch


path = r"C:\Users\joshu\Documents\BEMMedia\flat-lam2.stl"
root_path = r"C:\Users\joshu\Documents\BEMMedia"

reflector = load_scatterer(path,dz=-0.05)

board = TOP_BOARD

p = create_points(1,1,x=0,y=0,z=-0.04)

E = compute_E(reflector, p, board=board, path=root_path)

U_target = torch.tensor([-8e-6,]).to(device).to(DTYPE)

def MSE_gorkov(transducer_phases, points, board, targets, **objective_params):
    U = BEM_gorkov_analytical(transducer_phases, points, reflector, board, path=root_path)
    loss = torch.mean((targets-U)**2).unsqueeze_(0).real
    return loss

x = gradient_descent_solver(p, MSE_gorkov, board, log=True, targets=U_target, iters=100, lr=1e5)

print(BEM_gorkov_analytical(x, p, reflector, board, path=root_path))

abc = ABC(0.06)
Visualise(*abc, x, colour_functions=[propagate_BEM_pressure,], colour_function_args=[{'board':board,'scatterer':reflector, 'path':root_path}], res=(200,200))