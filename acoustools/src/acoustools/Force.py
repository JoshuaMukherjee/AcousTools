from acoustools.Gorkov import gorkov_fin_diff, get_finite_diff_points_all_axis
from acoustools.Utilities import forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, TRANSDUCERS, propagate, DTYPE
import acoustools.Constants as c
from acoustools.BEM import grad_H, grad_2_H, get_cache_or_compute_H, get_cache_or_compute_H_gradients
from acoustools.Mesh import translate, merge_scatterers, get_centres_as_points, get_areas, get_normals_as_points

import torch

def force_fin_diff(activations, points, axis="XYZ", stepsize = 0.000135156253,K1=None, K2=None,U_function=gorkov_fin_diff,U_fun_args={}):
    '''
    Returns the force on a particle using finite differences to approximate the derivative of the gor'kov potential\\
    `activations` Transducer hologram\\
    `points` Points to propagate to\\
    `axis` string containing `X`, `Y` or `Z` defining the axis to take into account eg `XYZ` considers all 3 axes and `YZ` considers only the y and z-axes\\
    `stepsize` stepsize to use for finite differences \\
    `K1` Value for K1 to be used in the gor'kov computation, see `Holographic acoustic elements for manipulation of levitated objects` for more information\\
    `K2` Value for K1 to be used in the gor'kov computation, see `Holographic acoustic elements for manipulation of levitated objects` for more information\\
    `U_function` The funvtion used to compute the gor'kov potential\\
    `U_fun_args` arguments for `U_function` \\
    Returns Force
    '''
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]

    fin_diff_points = get_finite_diff_points_all_axis(points, axis, stepsize)
    
    U_points = U_function(activations, fin_diff_points, axis=axis, stepsize=stepsize/10 ,K1=K1,K2=K2,**U_fun_args)
    U_grads = U_points[:,N:]
    split = torch.reshape(U_grads,(B,2,-1))
    
    F =  (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    return F


def compute_force(activations, points,board=TRANSDUCERS,return_components=False):
    '''
    Returns the force on a particle using the analytical derivative of the Gor'kov potential and the piston model\\  
    `activations` Transducer hologram\\
    `points` Points to propagate to\\
    `board` Transducers to use \\
    `return_components` If true returns force as one tensor otherwise returns Fx, Fy, Fz\\
    Returns force  
    '''
    
    F = forward_model_batched(points,transducers=board)
    Fx, Fy, Fz = forward_model_grad(points,transducers=board)
    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=board)
    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points,transducers=board)

    p   = torch.abs(F@activations)
    Px  = torch.abs(Fx@activations)
    Py  = torch.abs(Fy@activations)
    Pz  = torch.abs(Fz@activations)
    Pxx = torch.abs(Fxx@activations)
    Pyy = torch.abs(Fyy@activations)
    Pzz = torch.abs(Fzz@activations)
    Pxy = torch.abs(Fxy@activations)
    Pxz = torch.abs(Fxz@activations)
    Pyz = torch.abs(Fyz@activations)


    
    K1 = c.V / (4*c.p_0*c.c_0**2)
    K2 = 3*c.V / (4*(2*c.f**2 * c.p_0))

    single_sum = 2*K2*(Pz+Py+Pz)

    force_x = -1 * (2*p * (K1 * Px - K2*(Pxz+Pxy+Pxx)) - Px*single_sum)
    force_y = -1 * (2*p * (K1 * Py - K2*(Pyz+Pyy+Pxy)) - Py*single_sum)
    force_z = -1 * (2*p * (K1 * Pz - K2*(Pzz+Pyz+Pxz)) - Pz*single_sum)


    if return_components:
        return force_x, force_y, force_z
    else:
        force = torch.cat([force_x, force_y, force_z],2)
        return force
    
def get_force_axis(activations, points,board=TRANSDUCERS, axis=2):
    '''
    Returns the force in one axis on a particle using the analytical derivative of the Gor'kov potential and the piston model\\  
    `activations` Transducer hologram\\
    `points` Points to propagate to\\
    `board` Transducers to use \\
    `axis` Axis to take the force in\\
    Equivalent to `compute_force(activations, points,return_components=True)[axis]` \\
    Returns force  
    '''
    forces = compute_force(activations, points,return_components=True)
    force = forces[axis]

    return force


def force_mesh(activations, points, norms, areas, board, grad_function=forward_model_grad, grad_function_args={},F=None, Ax=None, Ay=None,Az=None):
    '''
    Returns the force on a mesh using a discritised version of Eq. 1 in `Acoustical boundary hologram for macroscopic rigid-body levitation`\\
    `activations` Transducer hologram\\
    `points` Points to propagate to\\
    `norms` The normals to the mesh faces\\
    `areas` The areas of the mesh points\\
    `board` Transducers to use \\
    `grad_function` The function to use to compute the gradient of pressure\\
    `grad_function_args` The argument to pass to `grad_function`\\
    `F` A precomputed forward propagation matrix, if None will be computed\\
    `Ax` The gradient of F wrt x, if None will be computed\\
    `Ay` The gradient of F wrt y, if None will be computed\\
    `Az` The gradient of F wrt z, if None will be computed\\
    Returns the force on each mesh element\\
    '''
    activations= activations
    p = propagate(activations,points,board,A=F)
    pressure = torch.abs(p)**2
    
    if Ax is None or Ay is None or Az is None:
        Ax, Ay, Az = grad_function(points=points, transducers=board, **grad_function_args)
    


    px = (Ax@activations).squeeze_(2).unsqueeze_(0)
    py = (Ay@activations).squeeze_(2).unsqueeze_(0)
    pz = (Az@activations).squeeze_(2).unsqueeze_(0)

    px[px.isnan()] = 0
    py[py.isnan()] = 0
    pz[pz.isnan()] = 0


    grad = torch.cat((px,py,pz),dim=1).to(DTYPE)
    grad_norm = torch.norm(grad,2,dim=1)**2
    
    k1 = 1/ (2*c.p_0*(c.c_0**2))
    k2 = 1/ (c.k**2)

    pressure = torch.unsqueeze(pressure,1).expand(-1,3,-1)  
    # print(pressure, k2*grad_norm)
    # force = -1/2*(k1 * (pressure * norms - k2*grad_norm*norms)) * areas # Old and maybe incorrect?
    force = -1/2 * k1 * (pressure - k2 * torch.abs(grad)**2) * norms * areas #Bks1. Page 299 for derivation
    force = torch.real(force) #Im(F) == 0 but needs to be complex till now for dtype compatability

    # print(torch.sgn(torch.sgn(force) * torch.log(torch.abs(force))) == torch.sgn(force))

    return force

def torque_mesh(activations, points, norms, areas, centre_of_mass, board,force=None, grad_function=forward_model_grad,grad_function_args={},F=None, Ax=None, Ay=None,Az=None):
    '''
    Returns the torque on a mesh using a discritised version of Eq. 1 in `Acoustical boundary hologram for macroscopic rigid-body levitation`\\
    `activations` Transducer hologram\\
    `points` Points to propagate to\\
    `norms` The normals to the mesh faces\\
    `areas` The areas of the mesh points\\
    `centre_of_mass` The position of the centre of mass of the mesh\\
    `board` Transducers to use \\
    `force` Precomputed force on the mesh faces, if None will be computed\\
    `grad_function` The function to use to compute the gradient of pressure\\
    `grad_function_args` The argument to pass to `grad_function`\\
    `F` A precomputed forward propagation matrix, if None will be computed\\
    `Ax` The gradient of F wrt x, if None will be computed\\
    `Ay` The gradient of F wrt y, if None will be computed\\
    `Az` The gradient of F wrt z, if None will be computed\\
    Returns the force on each mesh element\\        
    '''

    if force is None:
        force = force_mesh(activations, points, norms, areas, board,grad_function,grad_function_args,F=F, Ax=Ax, Ay=Ay, Az=Az)
    force = force.to(DTYPE)
    
    displacement = points - centre_of_mass
    displacement = displacement.to(DTYPE)
    torque = torch.linalg.cross(displacement,force,dim=1)

    return torch.real(torque)


def force_mesh_derivative(activations, points, norms, areas, board, scatterer,Hx = None, Hy=None, Hz=None, Haa=None):
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

    return Faa

def get_force_mesh_along_axis(start,end, activations, scatterers, board, mask=None, steps=200, path="Media",print_lines=False, use_cache=True, Hs = None, Hxs=None, Hys=None, Hzs=None):
    '''
    Computes the force on a mesh at each point from `start` to `end` with number of samples = `steps`  \\
    `start` The starting position\\
    `end` The ending position\\
    `activations` Transducer hologram\\
    `scatterers` First element is the mesh to move, rest is considered static reflectors \\
    `board` Transducers to use \\
    `mask` The mask to apply to filter force for only the mesh to move\\
    `steps` Number of steps to take from start to end\\
    `path` path to folder containing BEMCache/ \\
    `print_lines` if true prints messages detaling progress\\
    `use_cache` If true uses the cache system, otherwise computes H and does not save it\\
    `Hs` List of precomputed forward propagation matricies\\
    `Hxs` List of precomputed derivative of forward propagation matricies wrt x\\
    `Hys` List of precomputed derivative of forward propagation matricies wrt y\\
    `Hzs` List of precomputed derivative of forward propagation matricies wrt z\\
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
