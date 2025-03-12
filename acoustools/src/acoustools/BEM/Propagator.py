
import torch
from torch import Tensor

from vedo import Mesh

from acoustools.Utilities import TOP_BOARD
from acoustools.BEM.Forward_models import compute_E


def propagate_BEM(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None,
                  E:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False) ->Tensor:
    '''
    Propagates transducer phases to points using BEM\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :return pressure: complex pressure at points
    '''
    if board is None:
        board = TOP_BOARD

    if E is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E = compute_E(scatterer,points,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines)
    
    out = E@activations
    return out

def propagate_BEM_pressure(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:
                           Tensor|None=None,E:Tensor|None=None, path:str="Media",use_cache_H:bool=True, print_lines:bool=False) -> Tensor:
    '''
    Propagates transducer phases to points using BEM and returns absolute value of complex pressure\n
    Equivalent to `torch.abs(propagate_BEM(activations,points,scatterer,board,H,E,path))` \n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed 
    :param path: path to folder containing `BEMCache/ `
    
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    
    :return pressure: real pressure at points
    '''
    if board is None:
        board = TOP_BOARD

    point_activations = propagate_BEM(activations,points,scatterer,board,H,E,path,use_cache_H=use_cache_H,print_lines=print_lines)
    pressures =  torch.abs(point_activations)
    return pressures

 