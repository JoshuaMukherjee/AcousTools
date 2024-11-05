import torch
from torch import Tensor
from acoustools.Paths import interpolate_bezier, interpolate_bezier_acceleration


def optispline_min_distance_control_points(bezier:list[list[Tensor]], targets:list[Tensor],n:int, **params):
    '''
    OptiSpline objective to minimise distance between bezier spline and target points - minises MSE between target and points \n
    :param bezier: Bezier spline as list of (start, end, offset1, offset2) where offsets are from start 
    :param targets: Target points 
    :param n: number of points to sample
    :returns objective: objective value
    '''
    
    points = []
    for bez in bezier:
        bez_points = interpolate_bezier(*bez,n=n)
        points += bez_points
    points = torch.stack(points)
    return torch.mean((points - targets)**2)

def optispline_min_acceleration(bezier:list[list[Tensor]], targets:list[Tensor],n:int, **params):
    '''
    OptiSpline objective to minimise acceleration across the spline - minimises sum of absolute value of acceleration\n
    :param bezier: Bezier spline as list of (start, end, offset1, offset2) where offsets are from start 
    :param targets: Target points (ignored)
    :param n: number of points to sample
    :returns objective: objective value
    '''
    acels = []
    for bez in bezier:
        a = interpolate_bezier_acceleration(*bez,n)
    
        acels += a
    acels = torch.stack(acels)
    return torch.sum(torch.abs(acels))

def optispline_min_acceleration_position(bezier:list[list[Tensor]], targets:list[Tensor],n:int, **params):
    '''
    OptiSpline objective combines `optispline_min_acceleration` and `optispline_min_distance_control_points` - evaluated as `optispline_min_distance_control_points+alpha*optispline_min_acceleration` \n
    Requires alpha parameter in params \n
    :param bezier: Bezier spline as list of (start, end, offset1, offset2) where offsets are from start 
    :param targets: Target points 
    :param n: number of points to sample
    :returns objective: objective value
    '''
    alpha = params['alpha']
    return optispline_min_distance_control_points(bezier, targets,n, **params) + alpha * optispline_min_acceleration(bezier, targets,n, **params)