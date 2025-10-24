'''
Objective Functions to be used in Solver.gradient_descent_solver
Must have the signature (transducer_phases, points, board, targets, **objective_params) -> loss
'''

from acoustools.Utilities import propagate_abs, add_lev_sig
from acoustools.Gorkov import gorkov_analytical
from acoustools.BEM import propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Optimise.Constraints import sine_amplitude
import torch

from torch import Tensor

def propagate_abs_sum_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    Sum of the pressure of points
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    '''
    return torch.sum(propagate_abs(transducer_phases,points,board),dim=1)

def propagate_abs_sum_objective_BEM(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    Sum of the pressure of points
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    '''
    E = objective_params['E']
    scatterer = objective_params['scatterer']
    return torch.sum(propagate_BEM_pressure(transducer_phases,points,scatterer,board,E=E),dim=1).squeeze_(0)



def gorkov_analytical_sum_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor = None, **objective_params) -> Tensor:
    '''
    Sum of the Gor'kov potential of points
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    :param axis: axis to compute gor'kov potential in
    '''
    # transducer_phases = add_lev_sig(transducer_phases)
    
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    U = gorkov_analytical(transducer_phases, points, board, axis)

    return torch.sum(U,dim=1).squeeze_(1)

def gorkov_analytical_mean_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    Mean of the Gor'kov potential of points
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    :param axis: axis to compute gor'kov potential in
    '''
    
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    U = gorkov_analytical(transducer_phases, points, board, axis)
    return torch.mean(U,dim=1).squeeze_(1)

def gorkov_analytical_std_mean_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    Weighted sum of of standard deviation and mean of Gor'kov potential at points. \n
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    :param w: relative weight of standard deviation and mean
    :param axis: axis to compute gor'kov potential in
    '''
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    U = gorkov_analytical(transducer_phases, points, board, axis)
    
    w = objective_params["w"] if "w" in objective_params else 1

    return torch.mean(U,dim=1).squeeze(1) + w*torch.std(U,dim=1).squeeze(1)


def gorkov_trapping_stiffness_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    Objective from  Hirayama, R., Christopoulos, G., Martinez Plasencia, D., & Subramanian, S. (2022). \n
    High-speed acoustic holography with arbitrary scattering objects. \n
    In Sci. Adv (Vol. 8). https://doi.org/10.1126/sciadv.abn7614
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    :param w: relative weight of parts of objective
    :param axis: axis to compute gor'kov potential in
   
    '''
    t2 = add_lev_sig(transducer_phases)
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    w = objective_params["w"] if "w" in objective_params else 1e-4

    U = gorkov_analytical(t2, points, board, axis)

    return torch.sum(U + w*(torch.mean(U) - U)**2,dim=1).squeeze_(1)


def pressure_abs_gorkov_trapping_stiffness_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    weighted sum of `gorkov_trapping_stiffness_objective` and `propagate_abs_sum_objective`
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: <Unused>
    :param alpha: relative weight of parts of objective
    '''
    Ul = gorkov_trapping_stiffness_objective(transducer_phases, points, board, targets, **objective_params)
    pl = propagate_abs_sum_objective(transducer_phases, points, board, targets, **objective_params)

    alpha = objective_params["alpha"] if "alpha" in objective_params else 1

    return Ul + alpha*pl

def target_pressure_mse_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    MSE error of target pressure and true pressure
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target:target pressure
    '''
    p = propagate_abs(transducer_phases, points,board=board)
    l = torch.sum((p-targets)**2,dim=1)
    return l

def target_gorkov_mse_objective(transducer_phases: Tensor, points:Tensor, board:Tensor, targets:Tensor, **objective_params) -> Tensor:
    '''
    MSE error of target Gor'kov potential and true Gor'kov potential
    :param transducer_phases: Hologram
    :param points: Points
    :param board: Transducer board
    :param target: target gor'kov value
    '''
    # if "no_sig" not in objective_params:
    #     t2 = add_lev_sig(transducer_phases)
    # else:
    #     t2 = transducer_phases
    axis = objective_params["axis"] if "axis" in objective_params else "XYZ"
    U = gorkov_analytical(transducer_phases, points, board, axis)
    l = torch.mean((U-targets)**2,dim=1).squeeze_(1)
    
    return l

def target_gorkov_BEM_mse_objective(transducer_phases, points, board, targets, **objective_params):
    reflector = objective_params['reflector']
    root = objective_params['root']
    if 'dims' in objective_params:
        dims = objective_params['dims']
    else:
        dims = 'XYZ'
    U = BEM_gorkov_analytical(transducer_phases, points, reflector, board, path=root, dims=dims)
    loss = torch.mean((targets-U)**2).unsqueeze_(0).real
    return loss

def target_gorkov_BEM_mse_sine_objective(transducer_phases, points, board, targets, **objective_params):
    transducer_phases = sine_amplitude(transducer_phases)
    reflector = objective_params['reflector']
    root = objective_params['root']
    E = objective_params["E"] if "E" in objective_params else None
    if 'dims' in objective_params:
        dims = objective_params['dims']
    else:
        dims = 'XYZ'
    U = BEM_gorkov_analytical(transducer_phases, points, reflector, board, path=root, dims=dims, E=E)
    loss = torch.mean((targets-U)**2).unsqueeze_(0).real
    return loss