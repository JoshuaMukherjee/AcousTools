import torch
from acoustools.Utilities import device, propagate, forward_model_batched, forward_model_grad, TRANSDUCERS, DTYPE
import acoustools.Constants as c

from torch import Tensor
from types import FunctionType


def gorkov(activations: Tensor, points: Tensor,board:Tensor|None=None, axis:str="XYZ", V:float=c.V, p_ref=c.P_ref, k=c.k, transducer_radius = c.radius,
                        medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p, **params) -> Tensor:
    '''
    Use to compute the gorkov potential at a point. Alias for `src.acoustools.Gorkov.gorkov_analytical`
    '''
    return gorkov_analytical(activations, points, board, axis, V=V, p_ref=p_ref, k=k, transducer_radius=transducer_radius, 
                             medium_density=medium_density, medium_speed=medium_speed, particle_density=particle_density,particle_speed=particle_speed , **params)

def gorkov_analytical(activations: Tensor, points: Tensor,board:Tensor|None=None, axis:str="XYZ", board_norms = None,
                        V:float=c.V, p_ref=c.P_ref, k=c.k, transducer_radius = c.radius, 
                        medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p , **params) -> Tensor:
    '''
    Computes the Gorkov potential using analytical derivative of the piston model \n
    :param activation: The transducer activations to use 
    :param points: The points to compute the potential at
    :param board: The transducer boards to use
    :param axis: The axes to add points in as a string containing 'X', 'Y' and/or 'Z' eg 'XYZ' will use all three axis but 'YZ' will only add points in the YZ axis
    :param V: particle Volume
    :return: gorkov potential at each point
    ```Python
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    from acoustools.Gorkov import gorkov_analytical

    N=1
    B=1
    points = create_points(N,B)
    x = wgs(points)
    x = add_lev_sig(x)
    
    U_a = gorkov_analytical(x,points)

    print("Analytical",U_a.data.squeeze())
    ```
    '''

    if board is None:
        board = TRANSDUCERS

    Fx, Fy, Fz = forward_model_grad(points, transducers=board, p_ref=p_ref, k=k, transducer_radius=transducer_radius)
    F = forward_model_batched(points,board, p_ref=p_ref, k=k, transducer_radius=transducer_radius)
    
    p = torch.abs(F@activations)**2

    px = (Fx@activations) if 'X' in axis else 0
    py = (Fy@activations) if 'Y' in axis else 0
    pz = (Fz@activations) if 'Z' in axis else 0

    grad  = torch.cat((px,py,pz),dim=2)

    K1, K2 = get_gorkov_constants(V=V, c_0=medium_speed, c_p=particle_speed, p_0=medium_density, p_p=particle_density)
    g = (torch.sum(torch.abs(grad)**2, dim=2, keepdim=True))
    
    # K1 = 1/4*V*(1/(c.c_0**2*c.p_0) - 1/(c.c_p**2*c.p_p))
    # K2 = 3/4 * V * ((c.p_0 - c.p_p) / (c.angular_frequency**2 * c.p_0 * (c.p_0 * 2*c.p_p)))
    
    U = K1*p - K2*g
    return U


def get_gorkov_constants(V=c.V, p_0 = c.p_0, p_p=c.p_p, c_0=c.c_0, c_p=c.c_p, angular_frequency=c.angular_frequency ):
    '''
    Returns K1 and K2 for use in Gorkov computations, Uses the form shown in `Holographic acoustic elements for manipulation of levitated objects` \n
    :param: V: Particle Volume
    :param p_0: Density of medium
    :param p_p: Density of particle
    :param c_0: Speed of sound in medium
    :param c_p: speed of sound in particle
    :param angular_frequency: The angular frequency
    :returns K1, K2:

    '''
    #Derived Bk.3 Pg.91
    K1 = 1/4*V*(1/(c_0**2*p_0) - 1/(c_p**2*p_p)) 
    K2 = 3/4 * V * ((p_p - p_0) / (angular_frequency**2 * p_0 * (p_0 + 2*p_p))) 

    # K1 = V / (4*p_0*c_0**2)
    # K2 = 3*V / (4*(2*f**2 * p_0))

    return K1, K2

def gorkov_autograd(activations:Tensor, points:Tensor, K1:float|None=None, K2:float|None=None, V:float=c.V, p_ref=c.P_ref, k=c.k, transducer_radius = c.radius,
                     medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p,
                     retain_graph:bool=False,board:Tensor|None=None, board_norms = None,**params) -> Tensor:
    '''
    Computes the Gorkov potential using pytorch's autograd system\n
    :param activation: The transducer activations to use 
    :param points: The points to compute the potential at. if `None` will use `acoustools.Utilities.TRANSDUCERS`
    :param K1: The value for K1 in the Gorkov equation, if `None` will use `c.V / (4*c.p_0*c.c_0**2)`
    :param K2: The value for K2 in the Gorkov equation, if `None` will use `3*c.V / (4*(2*c.f**2 * c.p_0))`
    :param board: The transducer boards to use
    :param retain_graph: Value will be passed to autograd
    :return: gorkov potential at each point

    ```Python
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    from acoustools.Gorkov import gorkov_autograd

    N=1
    B=1
    points = create_points(N,B)
    x = wgs(points)
    x = add_lev_sig(x)
    
    U_ag = gorkov_autograd(x,points)

    print("Autograd", U_ag.data.squeeze())
    ```
    '''

    if board is None:
        board = TRANSDUCERS

    var_points = torch.autograd.Variable(points.data, requires_grad=True).to(device).to(DTYPE)

    B = points.shape[0]
    N = points.shape[2]
    
    if len(activations.shape) < 3:
        activations.unsqueeze_(0)    
    
    pressure = propagate(activations.to(DTYPE),var_points,board=board, p_ref=p_ref, k=k, transducer_radius=transducer_radius, norms=board_norms)
    pressure.backward(torch.ones((B,N))+0j, inputs=var_points, retain_graph=retain_graph)
    grad_pos = var_points.grad

    if K1 is None or K2 is None:
        K1_, K2_ = get_gorkov_constants(V=V, c_0=medium_speed, c_p=particle_speed, p_0=medium_density, p_p=particle_density)
        if K1 is None:
            K1 = K1_
        if K2 is None:
            K2 = K2_

    # K1 = 1/4*V*(1/(c.c_0**2*c.p_0) - 1/(c.c_p**2*c.p_p))
    # K2 = 3/4 * V * ((c.p_0 - c.p_p) / (c.angular_frequency**2 * c.p_0 * (c.p_0 * 2*c.p_p)))


    gorkov = K1 * torch.abs(pressure) **2 - K2 * torch.sum((torch.abs(grad_pos)**2),1)
    return gorkov


def gorkov_fin_diff(activations: Tensor, points:Tensor, axis:str="XYZ", stepsize:float = 0.000135156253,K1:float|None=None, K2:float|None=None,
                    prop_function:FunctionType=propagate,prop_fun_args:dict={}, board:Tensor|None=None, board_norms=None,
                    V=c.V, p_ref=c.P_ref, k=c.k, transducer_radius = c.radius,
                     medium_density=c.p_0, medium_speed = c.c_0, particle_density = c.p_p, particle_speed = c.c_p,) -> Tensor:
    '''
    Computes the Gorkov potential using finite differences to compute derivatives \n
    :param activation: The transducer activations to use 
    :param points: The points to compute the potential at
    :param axis: The axes to add points in as a string containing 'X', 'Y' and/or 'Z' eg 'XYZ' will use all three axis but 'YZ' will only add points in the YZ axis
    :param stepsize: The distance aroud points to add, default 0.000135156253
    :param K1: The value for K1 in the Gorkov equation, if `None` will use `c.V / (4*c.p_0*c.c_0**2)`
    :param K2: The value for K2 in the Gorkov equation, if `None` will use `3*c.V / (4*(2*c.f**2 * c.p_0))`
    :param prop_function: Function to use to compute pressure
    :param prop_fun_args: Arguments to pass to `prop_function`
    :param board: The transducer boards to use if `None` use `acoustools.Utilities.TRANSDUCERS`
    :param V: Particle Volume
    :return: gorkov potential at each point

    ```Python
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    from acoustools.Gorkov import gorkov_fin_diff

    N=1
    B=1
    points = create_points(N,B)
    x = wgs(points)
    x = add_lev_sig(x)
    
    U_fd = gorkov_fin_diff(x,points)

    print("Finite Differences",U_fd.data.squeeze())
    ```
    '''
    # torch.autograd.set_detect_anomaly(True)
    if board is None:
        board = TRANSDUCERS
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]

    
    if len(activations.shape) < 3:
        activations = torch.unsqueeze(activations,0).clone().to(device)

    fin_diff_points = get_finite_diff_points_all_axis(points, axis, stepsize)

    pressure_points = prop_function(activations, fin_diff_points,board=board,p_ref=p_ref, k=k, transducer_radius=transducer_radius, norms = board_norms,**prop_fun_args)
    # if len(pressure_points.shape)>1:
    # pressure_points = torch.squeeze(pressure_points,2)

    pressure = pressure_points[:,:N]
    pressure_fin_diff = pressure_points[:,N:]

    split = torch.reshape(pressure_fin_diff,(B,2, ((2*D))*N // 2))
    
    grad = (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    
    grad = torch.reshape(grad,(B,D,N))
    grad_abs_square = torch.pow(torch.abs(grad),2)
    grad_term = torch.sum(grad_abs_square,dim=1)

    if K1 is None or K2 is None:
        K1_, K2_ = get_gorkov_constants(V=V, c_0=medium_speed, c_p=particle_speed, p_0=medium_density, p_p=particle_density)
        if K1 is None:
            K1 = K1_
        if K2 is None:
            K2 = K2_
    
    # p_in =  torch.abs(pressure)
    p_in = torch.sqrt(torch.real(pressure) **2 + torch.imag(pressure)**2)
    if len(p_in.shape) > 2:
        p_in.squeeze_(2)
    # p_in = torch.squeeze(p_in,2)

    # K1 = 1/4*V*(1/(c.c_0**2*c.p_0) - 1/(c.c_p**2*c.p_p))
    # K2 = 3/4 * V * ((c.p_0 - c.p_p) / (c.angular_frequency**2 * c.p_0 * (c.p_0 * 2*c.p_p)))

    U = K1 * p_in**2 - K2 *grad_term
    
    return U


def get_finite_diff_points(points:Tensor , axis:Tensor, stepsize:float = 0.000135156253) -> Tensor:
    '''
    Gets points for finite difference calculations in one axis\n
    :param points: Points around which to find surrounding points
    :param axis: The axis to add points in
    :param stepsize: The distance aroud points to add, default 0.000135156253
    :return: points 
    '''
    #points = Bx3x4
    points_h = points.clone()
    points_neg_h = points.clone()
    points_h[:,axis,:] = points_h[:,axis,:] + stepsize
    points_neg_h[:,axis,:] = points_neg_h[:,axis,:] - stepsize

    return points_h, points_neg_h

def get_finite_diff_points_all_axis(points: Tensor,axis: str="XYZ", stepsize:float = 0.000135156253) -> Tensor:
    '''
    Gets points for finite difference calculations\\
    :param points: Points around which to find surrounding points\\
    :param axis: The axes to add points in as a string containing 'X', 'Y' and/or 'Z' eg 'XYZ' will use all three axis but 'YZ' will only add points in the YZ axis\\
    :param stepsize: The distance aroud points to add, default 0.000135156253\\
    :return: Points
    '''
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]
    fin_diff_points=  torch.zeros((B,3,((2*D)+1)*N)).to(device).to(DTYPE)
    fin_diff_points[:,:,:N] = points.clone()

    i = 2
    if "X" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 0, stepsize)
        fin_diff_points[:,:,N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h

        i += 1

    
    if "Y" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 1, stepsize)
        fin_diff_points[:,:,(i-1)*N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h
        i += 1
    
    if "Z" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 2, stepsize)
        fin_diff_points[:,:,(i-1)*N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h
        i += 1
    
    return fin_diff_points