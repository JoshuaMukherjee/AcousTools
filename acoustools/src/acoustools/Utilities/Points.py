from torch import Tensor
import torch

from acoustools.Utilities.Setup import device, DTYPE

def create_points(N:int=None,B:int=1,x:float|None=None,y:float|None=None,z:float|None=None, min_pos:float=-0.06, max_pos:float = 0.06) -> Tensor:
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

    its = [list, tuple]

    if N is None:
        if (type(x) in its and type(y) in its and type(z) in its ): 
            N = len(x)
        elif (type(x) == float and type(y) == float and type(z) == float) or (type(x) == int and type(y) == int and type(z) == int):
            N = 1
        else:
            raise ValueError("If N is not provided x,y and z need to be lists of points or single values")

    points = torch.zeros((B, 3, N), device=device)
    
    if x is not None:
        if type(x) in its:
            for i in range(N):
                points[:,0,i] = x[i]
        else:
            points[:,0,:] = x
    else:
        points[:,0,:].uniform_(min_pos, max_pos)
    
    if y is not None:
        if type(y) in its:
            for i in range(N):
                points[:,1,i] = y[i]
        else:
            points[:,1,:] = y
    else:
        points[:,1,:].uniform_(min_pos, max_pos)
    
    if z is not None:
        if type(z) in its:
            for i in range(N):
              points[:,2,i] = z[i]
        else:
            points[:,2,:] = z
    else:
        points[:,2,:].uniform_(min_pos, max_pos)


    return points.to(DTYPE)


def random_points_ring(N=1, origin:Tensor=None, min_r:float=0.01, max_r:float=0.06):

    theta = torch.rand((1,1,N)) * 2 * 3.1415
    phi = torch.rand((1,1,N)) * 2 * 3.1415
    
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    
    min_r_x = min_r_y = min_r_z = min_r

    max_r_x = max_r_y = max_r_z = max_r

    rx = torch.FloatTensor(1, 1, N).uniform_(min_r_x, max_r_x).to(device).to(DTYPE)
    ry = torch.FloatTensor(1, 1, N).uniform_(min_r_y, max_r_y).to(device).to(DTYPE)
    rz = torch.FloatTensor(1, 1, N).uniform_(min_r_z, max_r_z).to(device).to(DTYPE)
    
    r = (rx**2 + ry**2 + rz**2)**0.5
    
    x = r * sin_theta * cos_phi
    y = r * sin_theta * sin_phi
    z = r * cos_theta
    
    pts = torch.cat([x,y,z], dim=1)
    
    if origin is None: origin = create_points(N,1, 0, 0, 0)
    
    pts = pts + origin
    
    return pts

    
    
        
    