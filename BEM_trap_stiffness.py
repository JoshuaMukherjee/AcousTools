from acoustools.Utilities import *
from acoustools.BEM import BEM_gorkov_analytical
import torch
from torch import Tensor
from vedo import Mesh

#use compute stiffness code but replace compute_force with my code

def BEM_trap_stiffness(activations: Tensor, points: Tensor, scatterer: Mesh|None|str=None,
                       board: Tensor|None=None, H: Tensor|None=None, E: Tensor|None=None, 
                       delta: float=1e-6, dims: str='XYZ', **params)  -> Tensor:
    pass