from torch import Tensor
import torch

from acoustools.Utilities.Setup import device



def create_points(N:int,B:int=1,x:float|None=None,y:float|None=None,z:float|None=None, min_pos:float=-0.06, max_pos:float = 0.06) -> Tensor:
    '''
    Creates a random set of N points in B batches in shape `Bx3xN` \n
    :param N: Number of points per batch
    :param B: Number of Batches
    :param x: if not None all points will have this as their x position. Default: `None`
    :param y: if not None all points will have this as their y position. Default: `None`
    :param z: if not None all points will have this as their z position. Default: `None`
    :param min_pos: Minimum position
    :param max_pos: Maximum position
    ```
    from acoustools.Utilities import create_points
    p = create_points(N=3,B=1)
    ```
    '''
    points = torch.FloatTensor(B, 3, N).uniform_(min_pos,max_pos).to(device)
    
    if x is not None:
        if type(x) is float:
            points[:,0,:] = x
        elif type(x) is list:
            for i in range(N):
                points[:,0,i] = x[i]
    
    if y is not None:
        if type(y) is float:
            points[:,1,:] = y
        elif type(y) is list:
            for i in range(N):
                points[:,1,i] = y[i]
    
    if z is not None:
        if type(z) is float:
            points[:,2,:] = y
        elif type(z) is list:
            for i in range(N):
                points[:,2,i] = z[i]


    return points