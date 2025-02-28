from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points, device, DTYPE
from acoustools.Solvers import gradient_descent_solver
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical, propagate_BEM, get_cache_or_compute_H
from acoustools.Visualiser import Visualise, ABC
import acoustools.Constants as c

from acoustools.Optimise.Constraints import sine_amplitude

import vedo, torch

N=1

path = r"C:\Users\joshu\Documents\BEMMedia\flat-lam2.stl"

reflector = load_scatterer(path)

p = create_points(N,1,x=0.04,y=0,z=0.03)

board = TOP_BOARD

H = get_cache_or_compute_H(reflector, board,path=r"C:\Users\joshu\Documents\BEMMedia")

E = compute_E(reflector,points=p,board=board,path=r"C:\Users\joshu\Documents\BEMMedia",H=H)

def MSE_gorkov(transducer_phases, points, board, targets, **objective_params):
    transducer_phases = sine_amplitude(transducer_phases)
    U = BEM_gorkov_analytical(transducer_phases, points, reflector, board, path=path, H=H)
    loss = torch.mean((targets-U)**2).unsqueeze_(0).real
    return loss


U_target = torch.tensor([-7.5e-6,]).to(device).to(DTYPE)

x = gradient_descent_solver(p, MSE_gorkov, board, log=False, targets=U_target, iters=50, lr=1e4, constrains=sine_amplitude)


abc = ABC(0.06)
normal = (0,1,0)
origin = (0,0,0)

# Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":reflector,"board":TOP_BOARD,"path":r"C:\Users\joshu\Documents\BEMMedia"}],vmax=5000, show=True,res=[256,256],points=p)

print(propagate_BEM_pressure(x,p,reflector,TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia",H=H).item())
print(BEM_gorkov_analytical(x,p,reflector,TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia",H=H).item())
print(U_target.item())

Visualise(*abc, x, colour_functions=[propagate_BEM_pressure],
          colour_function_args=[{"scatterer":reflector,"board":TOP_BOARD,"path":r"C:\Users\joshu\Documents\BEMMedia",'H':H}], 
          show=True,res=[256,256],points=p)

