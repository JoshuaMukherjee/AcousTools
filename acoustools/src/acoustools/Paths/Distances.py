import torch
from torch import Tensor

import itertools, math


def distance(p1:Tensor, p2:Tensor) -> float:
    '''
    Computes the euclidian distance between two points\n
    :param p1: First point
    :param p2: Second point
    :return: Distance
    '''
    return torch.sqrt(torch.sum((p2 - p1)**2)).real


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

