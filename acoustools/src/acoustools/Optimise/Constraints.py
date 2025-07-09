'''
Constrains for Phases used in Solver.gradient_descent_solver
Must have signature phases, **params -> phases 
'''

import torch
from torch import Tensor

def constrain_phase_only(phases: Tensor, **params) -> Tensor:
    '''
    Normalises amplitude to 1
    :param phase: Phases
    :return: Hologram
    '''
    return phases / torch.abs(phases)

def constrant_normalise_amplitude(phases: Tensor, **params)-> Tensor:
    '''
    Constrains by dividing by `torch.max(torch.abs(phases))`
    :param phase: Phases
    :return: Hologram
    '''
    return phases / torch.max(torch.abs(phases))

def constrain_sigmoid_amplitude(phases: Tensor, **params)-> Tensor:
    '''
    Constrains by dividing by passing through signmoid
    :param phase: Phases
    :return: Hologram
    '''
    amplitudes = torch.abs(phases)
    norm_holo = phases / amplitudes
    con_amp = 0.5 * torch.sigmoid(amplitudes) + 1/2
    # print(torch.abs(norm_holo * sin_amp))
    return norm_holo * con_amp

def constrain_clamp_amp(phases: Tensor, **params)-> Tensor:
    '''
    Constrains by dividing by clamping
    :param phase: Phases
    :return: Hologram
    '''
    amplitudes = torch.abs(phases)
    norm_holo = phases / amplitudes
    clamp_amp = torch.clamp(amplitudes,min=0,max=1)
    return norm_holo * clamp_amp

def normalise_amplitude_normal(phases: Tensor, **params)-> Tensor:
    '''
    Constrains by dividing by z-score
    :param phase: Phases
    :return: Hologram
    '''
    amplitudes = torch.abs(phases)
    norm_holo = phases / amplitudes
    norm_dist_amp = (amplitudes - torch.min(amplitudes,dim=1,keepdim=True).values) / (torch.max(amplitudes,dim=1,keepdim=True).values - torch.min(amplitudes,dim=1,keepdim=True).values)
    # print(torch.abs(norm_holo * norm_dist_amp))
    return norm_holo * norm_dist_amp

def sine_amplitude(phases: Tensor, **params)-> Tensor:
    amplitudes = torch.abs(phases)
    sin_amp = torch.sin(amplitudes)
    angles = torch.angle(phases)
    return sin_amp * torch.exp(1j*angles)


def sine_amplitude_square(phases: Tensor, **params)-> Tensor:
    amplitudes = torch.abs(phases)
    sin_amp = torch.sin(amplitudes)**2
    angles = torch.angle(phases)
    return sin_amp * torch.exp(1j*angles)

def sine_amplitude_pi_square(phases: Tensor, **params)-> Tensor:
    amplitudes = torch.abs(phases)
    sin_amp = torch.sin(amplitudes * torch.pi/2)**2
    angles = torch.angle(phases)
    return sin_amp * torch.exp(1j*angles)

def sine_amplitude_pi(phases: Tensor, **params)-> Tensor:
    amplitudes = torch.abs(phases)
    sin_amp = torch.sin(amplitudes * torch.pi/2)
    angles = torch.angle(phases)
    return sin_amp * torch.exp(1j*angles)