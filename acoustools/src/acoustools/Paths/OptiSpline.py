
import torch, copy
from torch import Tensor
from types import FunctionType

from acoustools.Paths.Bezier import bezier_to_C1
from acoustools.Paths.Curves import Spline

def OptiSpline(spline:Spline, target_points:list[Tensor], objective: FunctionType, 
               n:int=20, C1:bool=True, optimiser:torch.optim.Optimizer=torch.optim.Adam, 
               lr: float=0.01, objective_params:dict={},iters:int=200,log=True, optimise_start:bool=True, get_intermediates = False ):
    '''
    Optimiser for AcousTools bezier Splines \n
    :param bezier: Bezier spline as list of (start, end, offset1, offset2) where offsets are from start 
    :param target_points: Target points 
    :param objective: Objective function
    :param n: number of points to sample
    :param C1: If True will enforce C1 continuity
    :param optimiser: Optimiser to use - default Adam
    :param lr: learning rate to use - default 0.01
    :param objective_params: Objectives to pass to objective function
    :param iters: iterations to optimise for
    :param log: If true will print objective value at each step
    :param optimise_start: If True will use the start position of the beziers as a optimisation parameter as well
    :returns bezier: Optimses curve

    '''
    params = []
    saved = []

    
    for curve in spline:
        ps = curve.get_OptiSpline_parameters(start=optimise_start)
        for p in ps:
            params.append(p.requires_grad_())

    optim = optimiser(params,lr)

    target_points = torch.stack(target_points)
    
    for epoch in range(iters):

        optim.zero_grad()       

        loss = objective( spline, target_points, n=n, **objective_params)
        if log: print(epoch, loss)

        loss.backward()
        optim.step()
        if C1: spline=bezier_to_C1(spline, get_points=False)
        

        if get_intermediates: saved.append(spline.clone())

    if C1: spline=bezier_to_C1(spline, get_points=False)

    if get_intermediates:
        return spline, saved

    return spline