from acoustools.Solvers import gradient_descent_solver
from acoustools.Optimise.Objectives import target_gorkov_mse_objective
from acoustools.Optimise.Constraints import constrain_clamp_amp

from acoustools.Utilities import create_points, TOP_BOARD
from acoustools.Visualiser import Visualise, ABC
from acoustools.Gorkov import gorkov_analytical
from acoustools.BEM import get_cache_or_compute_H, compute_H, BEM_gorkov_analytical, propagate_BEM_pressure
from acoustools.Mesh import load_scatterer

import torch
from torch import Tensor

board = TOP_BOARD

N = 1
positions = [0.01 - 0.001*i for i in range(N)]
axis = 2

path = "../BEMMedia"

reflector = load_scatterer(path+"/Flat-lam1.stl", dz=0)
H = get_cache_or_compute_H(reflector, board=board, path=path)

def gorkov_BEM_minimise(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    MSE error of target Gor'kov potential and true Gor'kov potential
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: target gor'kov value
    '''

   
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    U = BEM_gorkov_analytical(transducer_phases,points, reflector, board=board, H=H, path=path)
    l = -1 * torch.sum(U).reshape((1,))
    
    return l



for pos in positions:
        p = create_points(1,1,0,0,0)
        p[:,axis] = pos


        x = gradient_descent_solver(p,gorkov_BEM_minimise, 
                                            constrains=constrain_clamp_amp, lr=1e3, iters=500,log=True,
                                            objective_params={"no_sig":True},board=board)


        # U = gorkov_analytical(x, p, ).squeeze_().cpu().flatten().detach().numpy()
        U = BEM_gorkov_analytical(x,p, reflector, board=board, H=H, path=path)

        print(pos, U)

A,B,C = ABC(0.06, origin = [0,0,0.06])


Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{'scatterer':reflector,'board':board, 'path':path}], vmax=7000)
