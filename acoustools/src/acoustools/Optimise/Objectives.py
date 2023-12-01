'''
Objective Functions to be used in Solver.gradient_descent_solver
Must have the signature (transducer_phases, points, board, targets, **objective_params) -> loss
'''

from acoustools.Utilities import propagate_abs
import torch

def propagate_abs_sum_objective(transducer_phases, points, board, targets, **objective_params):
    return torch.sum(propagate_abs(transducer_phases,points,board))