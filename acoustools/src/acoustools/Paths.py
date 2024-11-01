import torch
import itertools
import math

from torch import Tensor

def get_numeral(numeral:int|str, A:Tensor, B:Tensor, C:Tensor) -> list[Tensor]:
    '''
    Get positions to draw a given numeral \n
    :param numeral: The number to generate a path for [1,9] 
    :param A: top left corner of the box to draw in
    :param B: bottom left corner of the box to draw in
    :param C: top right corner of the box to draw in
    :return: path

    '''
    if type(A) is not torch.Tensor or type(B) is not torch.Tensor or type(C) is not torch.Tensor:
        A = torch.Tensor(A)
        B = torch.Tensor(B)
        C = torch.Tensor(C)
    if int(numeral) == 1:
        return numeral_one(A,B,C)
    if int(numeral) == 2:
        return numeral_two(A,B,C)
    if int(numeral) == 3:
        return numeral_three(A,B,C)
    if int(numeral) == 4:
        return numeral_four(A,B,C)
    if int(numeral) == 5:
        return numeral_five(A,B,C)
    if int(numeral) == 6:
        return numeral_six(A,B,C)
    if int(numeral) == 7:
        return numeral_seven(A,B,C)
    if int(numeral) == 8:
        return numeral_eight(A,B,C)
    if int(numeral) == 9:
        return numeral_nine(A,B,C)

def numeral_one(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A + 0.5*AB + 0.1*AC)
    points.append(A + 0.5*AB + 0.9*AC)

    return points

def numeral_two(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points

def numeral_three(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.3 * AC)
    points.append(A+ 0.3*AB + 0.5 * AC)
    points.append(A+ 0.9*AB + 0.7 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)

    return points

def numeral_four(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points

def numeral_five(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.3 * AC)
    points.append(A+ 0.9*AB + 0.6 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)

    return points

def numeral_six(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)

    return points

def numeral_seven(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.5*AB + 0.1 * AC)
    points.append(A+ 0.5*AB + 0.9 * AC)

    return points

def numeral_eight(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.5*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)
    points.append(A+ 0.5*AB + 0.5 * AC)

    return points

def numeral_nine(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points


def distance(p1:Tensor, p2:Tensor) -> float:
    '''
    Computes the euclidian distance between two points
    :param p1: First point
    :param p2: Second point
    :return: Distance
    '''
    return torch.sqrt(torch.sum((p2 - p1)**2)).real
    
def interpolate_points(p1:Tensor, p2:Tensor, n:int)-> list[Tensor]:
    '''
    Interpolates `n` points between `p1` and `p2`
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


def total_distance(path: list[Tensor]):
    total_dist = 0
    distances = []
    for p1, p2 in itertools.pairwise(path):
        d = distance(p1,p2)
        total_dist +=  d
        distances.append(d)
    
    return total_dist, distances

def target_distance_to_n(total_dist, max_distance):
    n = total_dist / max_distance
    return math.ceil(n)

def interpolate_path(path: list[Tensor], n:int, return_distance:bool = False) -> list[Tensor]:
    '''
    Calls `interpolate_points on all adjacent pairs of points in path`
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

def interpolate_path_to_distance(path: list[Tensor], max_diatance:float=0.001) -> list[Tensor]:
    '''
    Calls `interpolate_points on all adjacent pairs of points in path to make distance < max_distance`
    :param max_diatance: max_distance between adjacent points
    :return: Path and optionally total distance
    '''
    points = []

    for i,(p1, p2) in enumerate(itertools.pairwise(path)):
        total_dist, distances = total_distance([p1,p2])
        n = math.ceil(total_dist / max_diatance)
        points += interpolate_points(p1, p2, n)
    
    return points


def interpolate_arc(start:Tensor, end:Tensor|None, origin:Tensor, n:int=100):

    #DIRECTION!!

    
    radius = torch.sqrt(torch.sum((start - origin)**2))

    start_vec = (start-origin)

    
    if end is not None:
        end_vec = (end-origin)
        cos = torch.dot(start_vec.squeeze(),end_vec.squeeze()) / (torch.linalg.vector_norm(start_vec.squeeze()) * torch.linalg.vector_norm(end_vec.squeeze()))
        angle = torch.acos(cos)
    else:
        end = start.clone()
        angle = 3.14159 * 2

    w = torch.cross(start_vec,end_vec,dim=1)
    u = start_vec 
    u/= torch.linalg.vector_norm(start_vec.squeeze())
    if  (w == 0).all():
        w += torch.ones_like(w) * 1e-10
    
    v = torch.cross(w,u,dim=1) 
    v /=  torch.linalg.vector_norm(v.squeeze())

    print(u)
    print(v)
    points = []
    for i in range(n):
            t = ((angle) / n) * i
            p = radius * (torch.cos(t)*u + torch.sin(t)*v) + origin
            points.append(p)

    return points


