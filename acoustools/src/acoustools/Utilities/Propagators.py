
import torch
from torch import Tensor

from acoustools.Utilities.Boards import TRANSDUCERS
from acoustools.Utilities.Utilities import is_batched_points
from acoustools.Utilities.Forward_models import forward_model_batched, forward_model
from acoustools.Utilities.Setup import device


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

