
from torch import Tensor
import torch

from acoustools.Utilities.Setup import device

def generate_gorkov_targets(N:int,B:int=1, max_val:float=-5, min_val:float=-5, log=True) -> Tensor:
    '''
    Generates a tensor of random negative Gor'kov potential values\n
    If `B=0` will return tensor with shape of `Nx1` else  will have shape `BxNx1`\n
    :param N: Number of values per batch
    :param B: Number of batches to produce
    :param max_val: Maximum exponent of the value that can be generated. Default: `0`
    :param min_val: Minimum exponent of the value that can be generated. Default: `-1e-4`
    :return: tensor of values
'''
   
    if B > 0:
        targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    else:
        targets = torch.FloatTensor(N,1).uniform_(min_val,max_val).to(device)
    

    targets =  -1*torch.pow(10,targets)
    
    return targets

def generate_pressure_targets(N:int,B:int=1, max_val:float=5000, min_val:float=3000) -> Tensor:
    '''
    Generates a tensor of random pressure values\\
    :param N: Number of values per batch
    :param B: Number of batches to produce
    :param max_val: Maximum value that can be generated. Default: `5000`
    :param min_val: Minimum value that can be generated. Default: `3000`
    Returns tensor of values
    '''
    targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    return targets