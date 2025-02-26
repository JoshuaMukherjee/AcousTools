from acoustools.Levitator import LevitatorController
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Mesh import load_scatterer
from acoustools.Utilities import create_points, TOP_BOARD, device, DTYPE
from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Objectives import target_gorkov_BEM_mse_objective

import vedo, torch

root = "../BEMMedia/" #Change to path to BEMMedia Folder
path = root+"flat-lam2.stl"

reflector = load_scatterer(path) #Change dz to be the position of the reflector

board = TOP_BOARD
p = create_points(1,1,0,0,0.05) #point at (0,0,0)

E = compute_E(reflector, p, board, path=root)

U_target = torch.tensor([-7.5e-6,]).to(device).to(DTYPE)


x = gradient_descent_solver(p, target_gorkov_BEM_mse_objective, board, log=False, targets=U_target, iters=50, 
                            lr=1e5, init_type='ones', objective_params={'reflector':reflector,'root':root})

pressure = propagate_BEM_pressure(x,p,reflector,E=E)
U = BEM_gorkov_analytical(x, p, reflector, board, path=root).item()

print(pressure)
print(U)

lev = LevitatorController(ids=(73,)) #Change to your board IDs
lev.levitate(x)
input("Press Enter to stop")
lev.disconnect()