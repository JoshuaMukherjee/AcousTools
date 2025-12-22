from acoustools.Gorkov import gorkov_fin_diff, get_finite_diff_points_all_axis, get_gorkov_constants
from acoustools.Utilities import forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, TRANSDUCERS, propagate, DTYPE
import acoustools.Constants as c

import torch
from torch import Tensor
from types import FunctionType


def force_fin_diff(activations:Tensor, points:Tensor, axis:str="XYZ", stepsize:float= 0.000135156253,K1:float|None=None, 
                   K2:float|None=None,U_function:FunctionType=gorkov_fin_diff,U_fun_args:dict={}, board:Tensor|None=None, V=c.V, p_ref=c.P_ref,
                    k=c.k, transducer_radius = c.radius,
                    medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p) -> Tensor:
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
    
    U_points = U_function(activations, fin_diff_points, axis=axis, stepsize=stepsize/10 ,K1=K1,K2=K2,**U_fun_args, board=board,V=V, 
                            p_ref=p_ref, k=k, transducer_radius=transducer_radius, 
                            medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density,particle_speed=particle_speed)
    U_grads = U_points[:,N:]
    split = torch.reshape(U_grads,(B,2,-1))
    # print(split)
    # print((split[:,0,:] - split[:,1,:]))

    # print()

    F = -1* (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    F = F.reshape(B,3,N).permute(0,2,1)
    return F

def compute_force(activations:Tensor, points:Tensor,board:Tensor|None=None,return_components:bool=False, V=c.V, p_ref=c.P_ref, 
                  transducer_radius=c.radius, k=c.k,
                 medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p) -> Tensor | tuple[Tensor, Tensor, Tensor]:
    '''
    Returns the force on a particle using the analytical derivative of the Gor'kov potential and the piston model\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param board: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param return_components: If true returns force as one tensor otherwise returns Fx, Fy, Fz
    :param V: Particle volume
    :return: force  
    '''

    #Bk.2 Pg.319

    if board is None:
        board = TRANSDUCERS
    
    F = forward_model_batched(points,transducers=board,p_ref=p_ref, transducer_radius=transducer_radius, k=k)
    Fx, Fy, Fz = forward_model_grad(points,transducers=board,p_ref=p_ref, transducer_radius=transducer_radius, k=k)
    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=board,p_ref=p_ref, transducer_radius=transducer_radius, k=k)
    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points,transducers=board,p_ref=p_ref, transducer_radius=transducer_radius, k=k)

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


    # grad_p = torch.stack([Px,Py,Pz], dim=2).squeeze(3)
    # grad_px = torch.stack([Pxx,Pxy,Pxz])
    # grad_py = torch.stack([Pxy,Pyy,Pyz])
    # grad_pz = torch.stack([Pxz,Pyz,Pzz])

    p_term_x= (p*Px.conj() + p.conj()*Px) 
    p_term_y= (p*Py.conj() + p.conj()*Py) 
    p_term_z= (p*Pz.conj() + p.conj()*Pz) 

    # px_term = Px*grad_px.conj() + Px.conj()*grad_px
    # py_term = Py*grad_py.conj() + Py.conj()*grad_py
    # pz_term = Pz*grad_pz.conj() + Pz.conj()*grad_pz

    # K1 = V / (4*c.p_0*c.c_0**2)
    # K2 = 3*V / (4*(2*c.f**2 * c.p_0))
    K1, K2 = get_gorkov_constants(V=V, c_0=medium_speed, c_p=particle_speed, p_0=medium_density, p_p=particle_density)

    # grad_U = K1 * p_term - K2 * (px_term + py_term + pz_term)

    grad_U_x = K1 * p_term_x - K2 * ((Pxx*Px.conj() + Px*Pxx.conj()) + (Pxy*Py.conj() + Py.conj()*Pxy) + (Pxz*Pz.conj() + Pxz.conj()*Pz))
    grad_U_y = K1 * p_term_y - K2 * ((Pxy*Px.conj() + Px*Pxy.conj()) + (Pyy*Py.conj() + Py.conj()*Pyy) + (Pyz*Pz.conj() + Pyz.conj()*Pz))
    grad_U_z = K1 * p_term_z - K2 * ((Pxz*Px.conj() + Px*Pxz.conj()) + (Pyz*Py.conj() + Py.conj()*Pyz) + (Pzz*Pz.conj() + Pxz.conj()*Pz))

    grad_U = torch.stack([grad_U_x, grad_U_y, grad_U_z])

    force = -(grad_U).real.squeeze(3).permute(1,2,0)
    
    if return_components:
        return force[:,:,0], force[:,:,1], force[:,:,2] 
    else:
        return force 

    
def get_force_axis(activations:Tensor, points:Tensor,board:Tensor|None=None, axis:int=2, transducer_radius=c.radius, k=c.k,
                 medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p) -> Tensor:
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
    forces = compute_force(activations, points,return_components=True, board=board, transducer_radius=transducer_radius, k=k, 
                           medium_density=medium_density, medium_speed=medium_speed,particle_density=particle_density, particle_speed=particle_speed)
    force = forces[axis]

    return force


def force_mesh(activations:Tensor, points:Tensor, norms:Tensor, areas:Tensor, board:Tensor, grad_function:FunctionType=forward_model_grad, 
               grad_function_args:dict={}, F_fun:FunctionType|None=forward_model_batched, F_function_args:dict={},
               F:Tensor|None=None, Ax:Tensor|None=None, Ay:Tensor|None=None,Az:Tensor|None=None,
               use_momentum:bool=False, return_components:bool=False, p_ref=c.P_ref, transducer_radius=c.radius, k=c.k, 
               medium_density = c.p_0, wave_speed = c.c_0, angular_frequency = c.angular_frequency ) -> Tensor:
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
    :param return_components: If True will return force, momentum_flux (force is still the total force)
    :return: the force on each mesh element
    '''

    if F is None:
        F = F_fun(points=points, transducers=board, p_ref=p_ref, transducer_radius=transducer_radius, k=k ,**F_function_args)
    p = propagate(activations,points,board,A=F, p_ref=p_ref, transducer_radius=transducer_radius, k=k)
    pressure_square = torch.abs(p)**2
    pressure_time_average = 1/2 * pressure_square

    # return pressure_time_average.expand((1,3,-1)), None 
    if Ax is None or Ay is None or Az is None:
        Ax, Ay, Az = grad_function(points=points, transducers=board, p_ref=p_ref, k=k, transducer_radius=transducer_radius, **grad_function_args)
    
    px = (Ax@activations).squeeze(2).unsqueeze(0)
    py = (Ay@activations).squeeze(2).unsqueeze(0)
    pz = (Az@activations).squeeze(2).unsqueeze(0)

    grad  = torch.cat((px,py,pz),dim=1) 
    velocity = grad /( 1j * medium_density * angular_frequency)

    
    k0 = 1/( medium_density * wave_speed**2)
    velocity_time_average = 1/2 * torch.sum(velocity * velocity.conj().resolve_conj(), dim=1, keepdim=True).real 

    # + velocity_time_average / velocity_time_average.max()
    # pressure_square / pressure_square.max()

    force = ( 0.5 * k0 * pressure_time_average - (medium_density / 2) * velocity_time_average) * norms

    if use_momentum:        
        momentum = medium_density/2 * (torch.sum(velocity * norms, dim=1, keepdim=True) * velocity.conj().resolve_conj()).real + 0j
        
        force += momentum 
    else:
        momentum = 0
    
    force *= -areas # *0.7
    # force = torch.real(force) #Im(F) == 0 but needs to be complex till now for dtype compatability
    # print(torch.sgn(torch.sgn(force) * torch.log(torch.abs(force))) == torch.sgn(force))

    if return_components: 
        return force, momentum
    
    return force

def torque_mesh(activations:Tensor, points:Tensor, norms:Tensor, areas:Tensor, centre_of_mass:Tensor, board:Tensor,force:Tensor|None=None, 
                grad_function:FunctionType=forward_model_grad,grad_function_args:dict={},F:Tensor|None=None, 
                Ax:Tensor|None=None, Ay:Tensor|None=None,Az:Tensor|None=None, transducer_radius=c.radius, k=c.k, 
               medium_density = c.p_0, wave_speed = c.c_0, angular_frequency = c.angular_frequency, p_ref=c.P_ref) -> Tensor:
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
        force = force_mesh(activations, points, norms, areas, board,grad_function,grad_function_args,F=F, Ax=Ax, Ay=Ay, Az=Az, 
                           p_ref=p_ref, k=k, wave_speed=wave_speed, medium_density=medium_density, transducer_radius=transducer_radius, angular_frequency=angular_frequency)
    force = force.to(DTYPE)
    
    displacement = points - centre_of_mass
    displacement = displacement.to(DTYPE)
    torque = torch.linalg.cross(displacement,force,dim=1)

    return torch.real(torque)


