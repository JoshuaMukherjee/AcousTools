
import torch
from torch import Tensor

from vedo import Mesh

from acoustools.Utilities import TOP_BOARD
from acoustools.BEM.Forward_models import compute_E
from acoustools.BEM.Gradients import BEM_forward_model_grad, BEM_laplacian
import acoustools.Constants as Constants

def propagate_BEM(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None,
                  E:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False, 
                  p_ref=Constants.P_ref,k:float=Constants.k, betas:float|Tensor = 0, alphas:float|Tensor=1, a=None,c=None, 
                  internal_points=None,smooth_distance=0, h=None, BM_alpha=None,
                  transducer_radius = Constants.radius) ->Tensor:
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
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param internal_points: The internal points to use for CHIEF based BEM
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM

    :return pressure: complex pressure at points
    '''
    if board is None:
        board = TOP_BOARD

    if E is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E = compute_E(scatterer,points,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines,p_ref=p_ref, k=k, betas=betas, alphas=alphas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius)
    
    out = E@activations
    return out

def propagate_BEM_pressure(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:
                           Tensor|None=None,E:Tensor|None=None, path:str="Media",use_cache_H:bool=True, print_lines:bool=False,p_ref=Constants.P_ref,k:float=Constants.k, betas = 0, alphas:float|Tensor=1, a=None,c=None, internal_points=None,smooth_distance=0, h=None, BM_alpha=None, transducer_radius = Constants.radius) -> Tensor:
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
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param internal_points: The internal points to use for CHIEF based BEM
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM

    
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    
    :return pressure: real pressure at points
    '''
    if board is None:
        board = TOP_BOARD

    point_activations = propagate_BEM(activations,points,scatterer,board,H,E,path,use_cache_H=use_cache_H,print_lines=print_lines,p_ref=p_ref, k=k, betas=betas, alphas=alphas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius)
    pressures =  torch.abs(point_activations)
    return pressures

def propagate_BEM_pressure_grad(activations: Tensor, points: Tensor,board: Tensor|None=None, scatterer:Mesh = None, 
                                path:str='Media', Fx=None, Fy=None, Fz=None, cat=True,p_ref=Constants.P_ref, k=Constants.k, transducer_radius = Constants.radius):
    '''
    Propagates a hologram to pressure gradient at points\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param Fx: The forward model to us for Fx, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fy: The forward model to us for Fy, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fz: The forward model to us for Fz, if None it is computed using `forward_model_grad`. Default:`None`
    :return: point velocity potential'
    '''
    
    if Fx is None or Fy is None or Fz is None:
        _Fx,_Fy,_Fz = BEM_forward_model_grad(points, scatterer ,board, p_ref=p_ref, k=k, transducer_radius=transducer_radius)
        if Fx is None: Fx = _Fx
        if Fy is None: Fy = _Fy
        if Fz is None: Fz = _Fz
    
    Px = Fx@activations
    Py = Fy@activations
    Pz = Fz@activations

    if cat: 
        grad = torch.cat([Px, Py, Pz], dim=2)
        return grad
    return Px, Py, Pz

def propagate_BEM_phase(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None,
                  E:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False,p_ref=Constants.P_ref,k:float=Constants.k, betas = 0, alphas:float|Tensor=1, a=None,c=None, internal_points=None,smooth_distance=0, h=None, BM_alpha=None, transducer_radius=Constants.radius) ->Tensor:
    '''
    Propagates transducer phases to phases at points using BEM\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param internal_points: The internal points to use for CHIEF based BEM
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM

    :return pressure: phase at points
    '''
    if board is None:
        board = TOP_BOARD

    if E is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E = compute_E(scatterer,points,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines, p_ref=p_ref, k=k, betas=betas, alphas=alphas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius)
    
    out = E@activations
    return torch.angle(out)


def propagate_BEM_laplacian(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None,
                  E_lap:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False,p_ref=Constants.P_ref,k:float=Constants.k, internal_points=None, transducer_radius=Constants.radius):
    '''
    Propagate transducer hologram to the laplacian of pressure at points
    
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :param k: wavenumber
    :param internal_points: The internal points to use for CHIEF based BEM

    :return pressure: laplacian at points
    '''
    
    if board is None:
        board = TOP_BOARD

    if E_lap is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E_lap = BEM_laplacian(points, scatterer ,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines, p_ref=p_ref, k=k, internal_points=internal_points, transducer_radius=transducer_radius)


    lap = E_lap @ activations

    return lap
    

def propagate_BEM_laplacian_abs(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None,
                  E_lap:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False,p_ref=Constants.P_ref,k:float=Constants.k, internal_points=None, transducer_radius=Constants.radius):
    '''
    Propagate transducer hologram to the absolute value of the laplacian of pressure at points

    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :param k: wavenumber
    :param internal_points: The internal points to use for CHIEF based BEM

    :return pressure: laplacian at points
    '''
    
    lap = propagate_BEM_laplacian(activations,points, scatterer ,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines, p_ref=p_ref, k=k, internal_points=internal_points, transducer_radius=transducer_radius, E_lap=E_lap)

    return torch.abs(lap)

def propagate_BEM_helmholtz(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None, E:Tensor|None=None,
                  E_lap:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False,p_ref=Constants.P_ref,k:float=Constants.k, internal_points=None, transducer_radius=Constants.radius):
    '''
    Computes the Helmholtz eq. at points given a hologram

    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :param k: wavenumber
    :param internal_points: The internal points to use for CHIEF based BEM

    :return pressure: laplacian at points
    '''
    
    lap = propagate_BEM_laplacian(activations,points, scatterer ,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines, p_ref=p_ref, k=k, internal_points=internal_points, transducer_radius=transducer_radius, E_lap=E_lap)
    pressure = propagate_BEM(activations=activations,points=points, scatterer=scatterer,board=board, H=H, E=E, use_cache_H=use_cache_H, path=path, p_ref=p_ref, k=k, internal_points=internal_points, transducer_radius=transducer_radius)


    return lap + k**2 * pressure


def propagate_BEM_helmholtz_abs(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None, E:Tensor|None=None,
                  E_lap:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False,p_ref=Constants.P_ref,k:float=Constants.k, internal_points=None, transducer_radius=Constants.radius):
    
    '''
    Computes the absolute value of the Helmholtz eq. at points given a hologram

    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :param k: wavenumber
    :param internal_points: The internal points to use for CHIEF based BEM

    :return pressure: laplacian at points
    '''
    
    helmholtz = propagate_BEM_helmholtz(activations,points, scatterer ,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines, p_ref=p_ref, k=k, internal_points=internal_points, transducer_radius=transducer_radius, E_lap=E_lap, E=E)
    
    return torch.abs(helmholtz)