import torch
from torch import Tensor
import itertools, math

from acoustools.Paths.Distances import total_distance
from acoustools.Utilities.Setup import device, DTYPE
from acoustools.Utilities.Points import create_points

def interpolate_path(path: list[Tensor], n:int, return_distance:bool = False) -> list[Tensor]:
    '''
    Calls `interpolate_points on all adjacent pairs of points in path`\n
    :param n: TOTAL number of points to interpolate (will be split between pairs)
    :param return_distance: if `True` will also return total distance
    :return: Path and optionally total distance
    '''
    points = []
    total_dist, distances = total_distance(path)


    for i,(p1, p2) in enumerate(itertools.pairwise(path)):
        d = distances[i]
        num = round((n * (d / total_dist )).item())
        points += interpolate_points(p1, p2, num)

    if return_distance:
        return points, total_dist
    return points

def interpolate_points(p1:Tensor, p2:Tensor, n:int)-> list[Tensor]:
    '''
    Interpolates `n` points between `p1` and `p2`\n
    :param p1: First point
    :param p2: Second point
    :param n: number of points to interpolate
    :return: Path
    '''
    if n > 0:
        vec = (p2 - p1) / n
        points = []
        for i in range(n):
            points.append(p1 + i * vec)
    else:
        return p1

    return points



def interpolate_path_to_distance(path: list[Tensor], max_diatance:float=0.001) -> list[Tensor]:
    '''
    Calls `interpolate_points on all adjacent pairs of points in path to make distance < max_distance` \n
    :param max_diatance: max_distance between adjacent points
    :return: Path and optionally total distance
    '''
    points = []

    for i,(p1, p2) in enumerate(itertools.pairwise(path)):
        total_dist, distances = total_distance([p1,p2])
        n = math.ceil(total_dist / max_diatance)
        points += interpolate_points(p1, p2, n)
    
    return points


def interpolate_arc(start:Tensor, end:Tensor|None=None, origin:Tensor=None, n:int=100, 
                    up:Tensor=torch.tensor([0,0,1.0]), anticlockwise:bool=False) -> list[Tensor]:
    
    '''
    Creates an arc between start and end with origin at origin\n
    :param start: Point defining start of the arc
    :param end: Point defining the end of the arc (this is not checked to lie on the arc - the arc will end at the same angle as arc)
    :param origin: Point defining origin of the arc
    :param n: number of points to interpolate along the arc. Default 100
    :param up: vector defining which way is 'up'. Default to positive z
    :param anticlickwise: If true will create anticlockwise arc. Otherwise will create clockwise arc
    :returns Points: List of points
    
    '''

    if origin is None:
        raise ValueError('Need to pass a value for origin')

    radius = torch.sqrt(torch.sum((start - origin)**2))

    start_vec = (start-origin)

    up = up.to(device).to(float)

    if end is not None:
        end_vec = (end-origin)
        cos = torch.dot(start_vec.squeeze(),end_vec.squeeze()) / (torch.linalg.vector_norm(start_vec.squeeze()) * torch.linalg.vector_norm(end_vec.squeeze()))
        angle = torch.acos(cos)
    else:
        end = start.clone() + 1e-10
        end_vec = (end-origin)
        angle = torch.tensor([3.14159 * 2]).to(device)

    w = torch.cross(start_vec,end_vec,dim=1).to(float)
    clockwise = torch.dot(w.squeeze(),up.squeeze())<0

    u = start_vec.to(float)
    u/= torch.linalg.vector_norm(start_vec.squeeze())
    if  (w == 0).all():
        w += torch.ones_like(w) * 1e-10
    
    v = torch.cross(w,u,dim=1) 
    v /=  torch.linalg.vector_norm(v.squeeze())

    if clockwise == anticlockwise: #Should be false
        angle = 2*3.14159 - angle
        direction= -1
    else:
        direction = 1

    points = []
    for i in range(n):
            t = direction * ((angle) / n) * i
            p = radius * (torch.cos(t)*u + torch.sin(t)*v) + origin
            points.append(p)

    return points


def interpolate_circle(origin:Tensor, radius:float=1.0, plane='xy', n:int=100) -> list[Tensor]:
    points = []
    for i in range(n):
        a = radius * math.sin((2*math.pi*i) / n)
        b = radius * math.cos((2*math.pi*i) / n)
        if plane == 'xy':
            p = create_points(1,1,a,b,0)
        elif plane == 'xz':
            p = create_points(1,1,a,0,b)
        elif plane == 'yz':
            p = create_points(1,1,0,a,b)
        points.append(p)
    return points