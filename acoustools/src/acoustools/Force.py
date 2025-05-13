from acoustools.Gorkov import gorkov_fin_diff, get_finite_diff_points_all_axis
from acoustools.Utilities import forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, TRANSDUCERS, propagate, DTYPE
import acoustools.Constants as c
from acoustools.BEM import grad_H, grad_2_H, get_cache_or_compute_H, get_cache_or_compute_H_gradients
from acoustools.Mesh import translate, merge_scatterers, get_centres_as_points, get_areas, get_normals_as_points

import torch
from torch import Tensor
from types import FunctionType
from vedo import Mesh

torch.set_printoptions(linewidth=400)

def force_fin_diff(activations:Tensor, points:Tensor, axis:str="XYZ", stepsize:float= 0.000135156253,K1:float|None=None, 
                   K2:float|None=None,U_function:FunctionType=gorkov_fin_diff,U_fun_args:dict={}, board:Tensor|None=None, V=c.V) -> Tensor:
    '''
    Returns the force on a particle using finite differences to approximate the derivative of the gor'kov potential\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param axis: string containing `X`, `Y` or `Z` defining the axis to take into account eg `XYZ` considers all 3 axes and `YZ` considers only the y and z-axes
    :param stepsize: stepsize to use for finite differences 
    :param K1: Value for K1 to be used in the gor'kov computation, see `Holographic acoustic elements for manipulation of levitated objects` for more information
    :param K2: Value for K1 to be used in the gor'kov computation, see `Holographic acoustic elements for manipulation of levitated objects` for more information
    :param U_function: The function used to compute the gor'kov potential
    :param U_fun_args: arguments for `U_function` 
    :param board: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :parm V: Particle volume
    :return: Force
    '''
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]

    if board is None:
        board = TRANSDUCERS

    fin_diff_points = get_finite_diff_points_all_axis(points, axis, stepsize)
    
    U_points = U_function(activations, fin_diff_points, axis=axis, stepsize=stepsize/10 ,K1=K1,K2=K2,**U_fun_args, board=board,V=V)
    U_grads = U_points[:,N:]
    split = torch.reshape(U_grads,(B,2,-1))

    
    F = -1* (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    F = F.reshape(B,3,N)
    return F

def compute_force(activations:Tensor, points:Tensor,board:Tensor|None=None,return_components:bool=False) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Returns the force on a particle using the analytical derivative of the Gor'kov potential and the piston model\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param board: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param return_components: If true returns force as one tensor otherwise returns Fx, Fy, Fz
    :return: force  
    '''

    #Bk.2 Pg.319

    if board is None:
        board = TRANSDUCERS
    
    F = forward_model_batched(points,transducers=board)
    Fx, Fy, Fz = forward_model_grad(points,transducers=board)
    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=board)
    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points,transducers=board)

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

    K1 = c.V / (4*c.p_0*c.c_0**2)
    K2 = 3*c.V / (4*(2*c.f**2 * c.p_0))

    grad_U = K1 * p_term - K2 * (px_term + py_term + pz_term)
    force = (-1 * grad_U).squeeze().real

    if return_components:
        return -1*force[0], -1*force[1], -1*force[2] 
    else:
        return -1*force 

    
def get_force_axis(activations:Tensor, points:Tensor,board:Tensor|None=None, axis:int=2) -> Tensor:
    '''
    Returns the force in one axis on a particle using the analytical derivative of the Gor'kov potential and the piston model \n
    Equivalent to `compute_force(activations, points,return_components=True)[axis]` \n 

    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param board: Transducers to use if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param axis: Axis to take the force in
    :return: force  
    '''
    if board is None:
        board = TRANSDUCERS
    forces = compute_force(activations, points,return_components=True, board=board)
    force = forces[axis]

    return force


def force_mesh(activations:Tensor, points:Tensor, norms:Tensor, areas:Tensor, board:Tensor, grad_function:FunctionType=forward_model_grad, 
               grad_function_args:dict={}, F_fun:FunctionType|None=forward_model_batched, F_function_args:dict={},
               F:Tensor|None=None, Ax:Tensor|None=None, Ay:Tensor|None=None,Az:Tensor|None=None,
               use_momentum:bool=False) -> Tensor:
    '''
    Returns the force on a mesh using a discritised version of Eq. 1 in `Acoustical boundary hologram for macroscopic rigid-body levitation`\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param norms: The normals to the mesh faces
    :param areas: The areas of the mesh points
    :param board: Transducers to use 
    :param grad_function: The function to use to compute the gradient of pressure
    :param grad_function_args: The argument to pass to `grad_function`
    :param F_fun: Function to compute F
    :param F_function_args:Fucntion to compute Grad F
    :param F: A precomputed forward propagation matrix, if `None` will be computed
    :param Ax: The gradient of `F` wrt x, if `None` will be computed
    :param Ay: The gradient of `F` wrt y, if `None` will be computed
    :param Az: The gradient of `F` wrt z, if `None` will be computed
    :param use_mpmentum: If true will add the term for momentum advection, for sound hard boundaries should be false
    :return: the force on each mesh element
    '''

    if F is None:
        F = F_fun(points=points,**F_function_args)
    p = propagate(activations,points,board,A=F)
    pressure_square = torch.abs(p)**2
    
    if Ax is None or Ay is None or Az is None:
        Ax, Ay, Az = grad_function(points=points, transducers=board, **grad_function_args)
    


    px = (Ax@activations).squeeze(2).unsqueeze(0)
    py = (Ay@activations).squeeze(2).unsqueeze(0)
    pz = (Az@activations).squeeze(2).unsqueeze(0)


    grad  = torch.cat((px,py,pz),dim=1)
    # grad_norm = torch.norm(grad,2,dim=1)**2
    grad_norm = torch.abs(px)**2 + torch.abs(py)**2 + torch.abs(pz)**2

    
    k1 = 1/ (4*c.p_0*(c.c_0**2))
    k2 = 1/ (c.k**2)

 
    force =  -1 * k1 * (pressure_square - k2 * grad_norm) * norms #Bk1. Page 299 for derivation, norm on pg 307

    if use_momentum:

        grad_normal = torch.sum(grad * norms, dim=1, keepdim=True)
        grad_conj = grad.conj().resolve_conj()
        momentum = -1 * 1/(2 * c.p_0 * c.angular_frequency**2) * (grad_normal * grad_conj).real
        force += momentum

    force *= areas

    # compressability = -1*k1*k2*(grad.conj().resolve_conj() @ (grad.mH @ norms) )* areas #Bk2. Pg 9
    # force += compressability.real 
    
    force = torch.real(force) #Im(F) == 0 but needs to be complex till now for dtype compatability

    # print(torch.sgn(torch.sgn(force) * torch.log(torch.abs(force))) == torch.sgn(force))

    return force

def torque_mesh(activations:Tensor, points:Tensor, norms:Tensor, areas:Tensor, centre_of_mass:Tensor, board:Tensor,force:Tensor|None=None, 
                grad_function:FunctionType=forward_model_grad,grad_function_args:dict={},F:Tensor|None=None, 
                Ax:Tensor|None=None, Ay:Tensor|None=None,Az:Tensor|None=None) -> Tensor:
    '''
    Returns the torque on a mesh using a discritised version of Eq. 1 in `Acoustical boundary hologram for macroscopic rigid-body levitation`\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param norms: The normals to the mesh faces
    :param areas: The areas of the mesh points
    :param centre_of_mass: The position of the centre of mass of the mesh
    :param board: Transducers to use 
    :param force: Precomputed force on the mesh faces, if `None` will be computed
    :param grad_function: The function to use to compute the gradient of pressure
    :param grad_function_args: The argument to pass to `grad_function`
    :param F: A precomputed forward propagation matrix, if `None` will be computed
    :param Ax: The gradient of F wrt x, if `None` will be computed
    :param Ay: The gradient of F wrt y, if `None` will be computed
    :param Az: The gradient of F wrt z, if `None` will be computed
    :return: the force on each mesh element
    '''

    if force is None:
        force = force_mesh(activations, points, norms, areas, board,grad_function,grad_function_args,F=F, Ax=Ax, Ay=Ay, Az=Az)
    force = force.to(DTYPE)
    
    displacement = points - centre_of_mass
    displacement = displacement.to(DTYPE)
    torque = torch.linalg.cross(displacement,force,dim=1)

    return torch.real(torque)


def force_mesh_derivative(activations, points, norms, areas, board, scatterer,Hx = None, Hy=None, Hz=None, Haa=None):
    '''
    @private
    '''
    print("Warning probably not correct...")
    if Hx is None or Hy is None or Hz is None:
        Hx, Hy, Hz, A, A_inv, Ax, Ay, Az = grad_H(points, scatterer, board, True)
    else:
        A, A_inv, Ax, Ay, Az = None, None, None, None, None

    if Haa is None:
        Haa = grad_2_H(points, scatterer, board, A, A_inv, Ax, Ay, Az)
    
    Ha = torch.stack([Hx,Hy,Hz],dim=1)

    Pa = Ha@activations
    Paa = Haa@activations

    Pa = Pa.squeeze(3)
    Paa = Paa.squeeze(3)

    k1 = 1/ (2*c.p_0*(c.c_0**2))
    k2 = 1/ (c.k**2)


    Faa =areas * k1 * (Pa * norms - 2*k2*norms*Pa*Paa)

    return Faa#

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
