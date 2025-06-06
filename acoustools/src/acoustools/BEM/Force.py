
from acoustools.BEM import BEM_forward_model_second_derivative_mixed, BEM_forward_model_second_derivative_unmixed, BEM_forward_model_grad, compute_E, get_cache_or_compute_H, get_cache_or_compute_H_gradients
from acoustools.Utilities import TRANSDUCERS
from acoustools.Force import force_mesh
from acoustools.Mesh import load_scatterer, get_centres_as_points, get_normals_as_points, get_areas, scale_to_diameter, centre_scatterer, translate, merge_scatterers, get_edge_data

import acoustools.Constants as c

from torch import Tensor
import torch

from vedo import Mesh

def BEM_compute_force(activations:Tensor, points:Tensor,board:Tensor|None=None,return_components:bool=False, V=c.V, scatterer:Mesh=None, 
                  H:Tensor=None, path:str="Media") -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Returns the force on a particle using the analytical derivative of the Gor'kov potential and BEM\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param board: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param return_components: If true returns force as one tensor otherwise returns Fx, Fy, Fz
    :param V: Particle volume
    :param scatterer: Scatterer to use
    :param H: H to use, will load/compute if None
    :param path: Path to folder containing BEMCache
    :return: force  
    '''

    #Bk.2 Pg.319

    if board is None:
        board = TRANSDUCERS
    
    F = compute_E(scatterer=scatterer,points=points,board=board,H=H, path=path)
    Fx, Fy, Fz = BEM_forward_model_grad(points,transducers=board,scatterer=scatterer,H=H, path=path)
    Fxx, Fyy, Fzz = BEM_forward_model_second_derivative_unmixed(points,transducers=board,scatterer=scatterer,H=H, path=path)
    Fxy, Fxz, Fyz = BEM_forward_model_second_derivative_mixed(points,transducers=board,scatterer=scatterer,H=H, path=path)

    p   = (F@activations)
    Px  = (Fx@activations)
    Py  = (Fy@activations)
    Pz  = (Fz@activations)
    Pxx = (Fxx@activations)
    Pyy = (Fyy@activations)
    Pzz = (Fzz@activations)
    Pxy = (Fxy@activations)
    Pxz = (Fxz@activations)
    Pyz = (Fyz@activations)


    grad_p = torch.stack([Px,Py,Pz])
    grad_px = torch.stack([Pxx,Pxy,Pxz])
    grad_py = torch.stack([Pxy,Pyy,Pyz])
    grad_pz = torch.stack([Pxz,Pyz,Pzz])


    p_term = p*grad_p.conj() + p.conj()*grad_p

    px_term = Px*grad_px.conj() + Px.conj()*grad_px
    py_term = Py*grad_py.conj() + Py.conj()*grad_py
    pz_term = Pz*grad_pz.conj() + Pz.conj()*grad_pz

    K1 = V / (4*c.p_0*c.c_0**2)
    K2 = 3*V / (4*(2*c.f**2 * c.p_0))

    grad_U = K1 * p_term - K2 * (px_term + py_term + pz_term)
    force = -1*(grad_U).squeeze().real

    if return_components:
        return force[0], force[1], force[2] 
    else:
        return force 


def force_mesh_surface(activations:Tensor, scatterer:Mesh=None, board:Tensor|None=None,
                       return_components:bool=False, sum_elements = True,
                       H:Tensor=None, diameter=c.wavelength*2,
                       path:str="Media", surface_path:str = "/Sphere-solidworks-lam2.stl",
                       surface:Mesh|None=None, use_cache_H:bool=True) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    
    if surface is None:
        surface = load_scatterer(path+surface_path)
        scale_to_diameter(surface,diameter, reset=False, origin=True)
        centre_scatterer(surface)
        #Need to translate to object centre -> TODO 


    points = get_centres_as_points(surface)
    norms = get_normals_as_points(surface)
    areas = get_areas(surface)
    

    E,F,G,H = compute_E(scatterer, points, board,path=path, H=H, return_components=True, use_cache_H=use_cache_H)
    
    force = force_mesh(activations, points,norms,areas,board=board,F=E, use_momentum=True,
                    grad_function=BEM_forward_model_grad, grad_function_args={'scatterer':scatterer,
                                                                                'H':H,
                                                                                'path':path})
    if sum_elements: force=torch.sum(force, dim=2)

    if return_components:
        return (force[:,0]), (force[:,1]), (force[:,2])
    return force


def get_force_mesh_along_axis(start:Tensor,end:Tensor, activations:Tensor, scatterers:list[Mesh], board:Tensor, mask:Tensor|None=None, steps:int=200, 
                              path:str="Media",print_lines:bool=False, use_cache:bool=True, 
                              Hs:Tensor|None = None, Hxs:Tensor|None=None, Hys:Tensor|None=None, Hzs:Tensor|None=None) -> tuple[list[Tensor],list[Tensor],list[Tensor]]:
    '''
    Computes the force on a mesh at each point from `start` to `end` with number of samples = `steps`  \n
    :param start: The starting position
    :param end: The ending position
    :param activations: Transducer hologram
    :param scatterers: First element is the mesh to move, rest is considered static reflectors 
    :param board: Transducers to use 
    :param mask: The mask to apply to filter force for only the mesh to move
    :param steps: Number of steps to take from start to end
    :param path: path to folder containing BEMCache/ 
    :param print_lines: if true prints messages detaling progress
    :param use_cache: If true uses the cache system, otherwise computes H and does not save it
    :param Hs: List of precomputed forward propagation matricies
    :param Hxs: List of precomputed derivative of forward propagation matricies wrt x
    :param Hys: List of precomputed derivative of forward propagation matricies wrt y
    :param Hzs: List of precomputed derivative of forward propagation matricies wrt z
    :return: list for each axis of the force at each position
    '''
    # if Ax is None or Ay is None or Az is None:
    #     Ax, Ay, Az = grad_function(points=points, transducers=board, **grad_function_args)
    direction = (end - start) / steps

    translate(scatterers[0], start[0].item() - direction[0].item(), start[1].item() - direction[1].item(), start[2].item() - direction[2].item())
    scatterer = merge_scatterers(*scatterers)

    points = get_centres_as_points(scatterer)
    if mask is None:
        mask = torch.ones(points.shape[2]).to(bool)

    Fxs = []
    Fys = []
    Fzs = []

    for i in range(steps+1):
        if print_lines:
            print(i)
        
        
        translate(scatterers[0], direction[0].item(), direction[1].item(), direction[2].item())
        scatterer = merge_scatterers(*scatterers)

        points = get_centres_as_points(scatterer)
        areas = get_areas(scatterer)
        norms = get_normals_as_points(scatterer)

        if Hs is None:
            H = get_cache_or_compute_H(scatterer, board, path=path, print_lines=print_lines, use_cache_H=use_cache)
        else:
            H = Hs[i]
        
        if Hxs is None or Hys is None or Hzs is None:
            Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board, path=path, print_lines=print_lines, use_cache_H_grad=use_cache)
        else:
            Hx = Hxs[i]
            Hy = Hys[i]
            Hz = Hzs[i]
        

        force = force_mesh(activations, points, norms, areas, board, F=H, Ax=Hx, Ay=Hy, Az=Hz)

        force = torch.sum(force[:,:,mask],dim=2).squeeze()
        Fxs.append(force[0])
        Fys.append(force[1])
        Fzs.append(force[2])
        
        # print(i, force[0].item(), force[1].item(),force[2].item())
    return Fxs, Fys, Fzs
