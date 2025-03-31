
import torch
from torch import Tensor

from acoustools.Utilities.Boards import TRANSDUCERS
from acoustools.Utilities.Utilities import is_batched_points
from acoustools.Utilities.Forward_models import forward_model_batched, forward_model
from acoustools.Utilities.Piston_model_gradients import forward_model_grad
from acoustools.Utilities.Setup import device
import acoustools.Constants as c

from types import FunctionType


    
def propagate(activations: Tensor, points: Tensor,board: Tensor|None=None, A:Tensor|None=None) -> Tensor:
    '''
    Propagates a hologram to target points\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param A: The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`
    :return: point activations

    ```Python
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, propagate

    p = create_points(2,1)
    x = iterative_backpropagation(p)
    
    p = p.squeeze(0)
    x = iterative_backpropagation(p)
    print(propagate(x,p))
    ```
    '''
    if board is None:
        board  = TRANSDUCERS
    batch = is_batched_points(points)

    if A is None:
        if len(points.shape)>2:
            A = forward_model_batched(points,board).to(device)
        else:
            A = forward_model(points,board).to(device)
    prop = A@activations
    if batch:
        prop = torch.squeeze(prop, 2)
    return prop

def propagate_abs(activations: Tensor, points: Tensor,board:Tensor|None=None, A:Tensor|None=None, A_function:FunctionType=None, A_function_args:dict={}) -> Tensor:
    '''
    Propagates a hologram to target points and returns pressure - Same as `torch.abs(propagate(activations, points,board, A))`\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param A: The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`
    :return: point pressure

    ```Python
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(2,1)
    x = iterative_backpropagation(p)
    
    p = p.squeeze(0)
    x = iterative_backpropagation(p)
    print(propagate_abs(x,p))
    ```
    '''
    if board is None:
        board = TRANSDUCERS
    if A_function is not None:
        A = A_function(points, board, **A_function_args)

    out = propagate(activations, points,board,A=A)
    return torch.abs(out)

def propagate_phase(activations:Tensor, points:Tensor,board:Tensor|None=None, A:Tensor|None=None) -> Tensor:
    '''
    Propagates a hologram to target points and returns phase - Same as `torch.angle(propagate(activations, points,board, A))`\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param A: The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`
    :return: point phase

    ```Python
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, propagate_phase

    p = create_points(2,1)
    x = iterative_backpropagation(p)
    
    p = p.squeeze(0)
    x = iterative_backpropagation(p)
    print(propagate_phase(x,p))
    ```
    '''
    if board is None:
        board = TRANSDUCERS
    out = propagate(activations, points,board,A=A)
    return torch.angle(out)


def propagate_velocity_potential(activations: Tensor, points: Tensor,board: Tensor|None=None, A:Tensor|None=None, 
                                 density = c.p_0, angular_frequency = c.angular_frequency) -> Tensor:
    '''
    Propagates a hologram to velocity potential at points\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param A: The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`
    :return: point velocity potential'
    '''

    pressure = propagate(activations, points, board, A=A)
    velocity_potential = pressure / (1j * density * angular_frequency)

    return velocity_potential

def propagate_pressure_grad(activations: Tensor, points: Tensor,board: Tensor|None=None, Fx=None, Fy=None, Fz=None):
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
        _Fx,_Fy,_Fz = forward_model_grad(points, board)
        if Fx is None: Fx = _Fx
        if Fy is None: Fy = _Fy
        if Fz is None: Fz = _Fz

    Px = Fx@activations
    Py = Fy@activations
    Pz = Fz@activations

    return Px, Py, Pz


def propagate_velocity(activations: Tensor, points: Tensor,board: Tensor|None=None, Fx=None, Fy=None, Fz=None, 
                                 density = c.p_0, angular_frequency = c.angular_frequency):
    '''
    Propagates a hologram to velocity at points\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param Fx: The forward model to us for Fx, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fy: The forward model to us for Fy, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fz: The forward model to us for Fz, if None it is computed using `forward_model_grad`. Default:`None`
    :return: point velocity potential'
    '''
    
    pressure_grads = propagate_pressure_grad(activations, points,board, Fx, Fy, Fz)
    alpha = 1/(1j * density * angular_frequency)
    velocity = [alpha * i for i in pressure_grads]
    return velocity

def propagate_velocity_real(activations: Tensor, points: Tensor,board: Tensor|None=None, Fx=None, Fy=None, Fz=None, 
                                 density = c.p_0, angular_frequency = c.angular_frequency):
    '''
    Propagates a hologram to velocity's real component at points\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param Fx: The forward model to us for Fx, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fy: The forward model to us for Fy, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fz: The forward model to us for Fz, if None it is computed using `forward_model_grad`. Default:`None`
    :return: point velocity potential'
    '''
    
    return [i.real for i in propagate_velocity(activations, points,board, Fx, Fy, Fz, density, angular_frequency)]

def propagate_speed(activations: Tensor, points: Tensor,board: Tensor|None=None, Fx=None, Fy=None, Fz=None, 
                                 density = c.p_0, angular_frequency = c.angular_frequency):
    '''
    Propagates a hologram to speed at points\n
    :param activations: Hologram to use
    :param points: Points to propagate to
    :param board: The Transducer array, default two 16x16 arrays
    :param Fx: The forward model to us for Fx, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fy: The forward model to us for Fy, if None it is computed using `forward_model_grad`. Default:`None`
    :param Fz: The forward model to us for Fz, if None it is computed using `forward_model_grad`. Default:`None`
    :return: point velocity potential'
    '''
    
    velocity = propagate_velocity(activations, points,board, Fx, Fy, Fz, density, angular_frequency)
    speeds = []
    for vel in velocity:
        speeds.append(torch.abs(vel))
    speed = 0
    for spd in speeds:
        speed += torch.square(spd)
    speed = torch.sqrt(speed)
    return speed