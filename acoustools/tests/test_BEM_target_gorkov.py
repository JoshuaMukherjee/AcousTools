from acoustools.Utilities import create_points, TOP_BOARD, device, DTYPE, create_board
from acoustools.Mesh import load_scatterer
from acoustools.BEM import BEM_gorkov_analytical, compute_E, propagate_BEM_pressure

from acoustools.Solvers import gradient_descent_solver

from acoustools.Visualiser import Visualise, ABC

import torch


# path = r"C:\Users\joshu\Documents\BEMMedia\flat-lam2.stl"
# root_path = r"C:\Users\joshu\Documents\BEMMedia"
path = 'flat-lam2.stl'
root_path = '../BEMMedia/'

reflector = load_scatterer(root_path+path,dz=-0.05)

board = TOP_BOARD
# board = create_board(17,0.06)

p = create_points(1,1,x=0,y=0,z=-0.04)

E = compute_E(reflector, p, board=board, path=root_path)

U_target = torch.tensor([-1e-6,]).to(device).to(DTYPE)

def MSE_gorkov(transducer_phases, points, board, targets, **objective_params):
    U = BEM_gorkov_analytical(transducer_phases, points, reflector, board, path=root_path)
    loss = torch.mean((targets-U)**2).unsqueeze_(0).real
    return loss

x = gradient_descent_solver(p, MSE_gorkov, board, log=False, targets=U_target, iters=50, lr=1e5, init_type='trap')

print(U_target.item().real)
print(BEM_gorkov_analytical(x, p, reflector, board, path=root_path).item())

exit()

abc = ABC(0.06)
Visualise(*abc, x,points=p, colour_functions=[propagate_BEM_pressure,], colour_function_args=[{'board':board,'scatterer':reflector, 'path':root_path}], res=(200,200))