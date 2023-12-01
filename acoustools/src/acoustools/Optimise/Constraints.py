'''
Constrains for Phases used in Solver.gradient_descent_solver
Must have signature phases, **params -> phases 
'''

import torch

def constrain_phase_only(phases, **params):
    return phases / torch.abs(phases)

def constrant_normalise_amplitude(phases, **params):
    return phases / torch.max(torch.abs(phases))