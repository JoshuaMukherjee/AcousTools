'''
Constrains for Phases used in Solver.gradient_descent_solver
Must have signature phases, **params -> phases 
'''

import torch

def constrain_phase_only(phases, **params):
    return phases / torch.abs(phases)

def constrant_normalise_amplitude(phases, **params):
    return phases / torch.max(torch.abs(phases))

def constrain_sigmoid_amplitude(phases, **params):
    amplitudes = torch.abs(phases)
    norm_holo = phases / amplitudes
    con_amp = 0.5 * torch.sigmoid(amplitudes) + 1/2
    # print(torch.abs(norm_holo * sin_amp))
    return norm_holo * con_amp

def constrain_clamp_amp(phases, **params):
    amplitudes = torch.abs(phases)
    norm_holo = phases / amplitudes
    clamp_amp = torch.clamp(amplitudes,min=0,max=1)
    return norm_holo * clamp_amp

def normalise_amplitude_normal(phases, **params):
    amplitudes = torch.abs(phases)
    norm_holo = phases / amplitudes
    norm_dist_amp = (amplitudes - torch.min(amplitudes,dim=1,keepdim=True).values) / (torch.max(amplitudes,dim=1,keepdim=True).values - torch.min(amplitudes,dim=1,keepdim=True).values)
    # print(torch.abs(norm_holo * norm_dist_amp))
    return norm_holo * norm_dist_amp