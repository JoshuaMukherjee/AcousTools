'''
Objective Functions to be used in Solver.gradient_descent_solver
Must have the signature (transducer_phases, points, board, targets, **objective_params) -> loss
'''

from acoustools.Utilities import propagate_abs, add_lev_sig
from acoustools.Gorkov import gorkov_analytical
import torch

def propagate_abs_sum_objective(transducer_phases, points, board, targets, **objective_params):
    return torch.sum(propagate_abs(transducer_phases,points,board),dim=1)


def gorkov_analytical_sum_objective(transducer_phases, points, board, targets, **objective_params):
    transducer_phases = add_lev_sig(transducer_phases)
    
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    U = gorkov_analytical(transducer_phases, points, board, axis)

    return torch.sum(U,dim=1).squeeze_(1)

def gorkov_trapping_stiffness_objective(transducer_phases, points, board, targets, **objective_params):
    '''
    Adapted from: \\
    Hirayama, R., Christopoulos, G., Martinez Plasencia, D., & Subramanian, S. (2022). \\
    High-speed acoustic holography with arbitrary scattering objects. \\
    In Sci. Adv (Vol. 8). https://www.science.org
    '''
    t2 = add_lev_sig(transducer_phases)
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    w = objective_params["w"] if "w" in objective_params else 1e-4

    U = gorkov_analytical(t2, points, board, axis)

    return torch.sum(U + w*(torch.mean(U) - U)**2,dim=1).squeeze_(1)



