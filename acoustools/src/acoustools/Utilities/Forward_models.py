import torch
from torch import Tensor

from acoustools.Utilities.Boards import TRANSDUCERS
from acoustools.Utilities.Utilities import is_batched_points
import acoustools.Constants as Constants



def forward_model(points:Tensor, transducers:Tensor|None = None) -> Tensor:
    '''
    Create the piston model forward propagation matrix for points and transducers\\
    :param points: Point position to compute propagation to \\
    :param transducers: The Transducer array, default two 16x16 arrays \\
    Returns forward propagation matrix \\
    '''
    if transducers is None:
        transducers = TRANSDUCERS

    if is_batched_points(points):
        return forward_model_batched(points, transducers)
    else:
        return forward_model_unbatched(points, transducers)

def forward_model_unbatched(points, transducers = TRANSDUCERS):
    '''
    @private
    Compute the piston model for acoustic wave propagation NOTE: Unbatched, use `forward_model_batched` for batched computation \\
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns forward propagation matrix \\
    Written by Giorgos Christopoulos, 2022
    '''
    
    m=points.size()[1]
    n=transducers.size()[0]
    
    transducers_x=torch.reshape(transducers[:,0],(n,1))
    transducers_y=torch.reshape(transducers[:,1],(n,1))
    transducers_z=torch.reshape(transducers[:,2],(n,1))


    points_x=torch.reshape(points[0,:],(m,1))
    points_y=torch.reshape(points[1,:],(m,1))
    points_z=torch.reshape(points[2,:],(m,1))
    
    dx = (transducers_x.T-points_x) **2
    dy = (transducers_y.T-points_y) **2
    dz = (transducers_z.T-points_z) **2

    distance=torch.sqrt(dx+dy+dz)
    planar_distance=torch.sqrt(dx+dy)

    bessel_arg=Constants.k*Constants.radius*torch.divide(planar_distance,distance) #planar_dist / dist = sin(theta)

    directivity=1/2-torch.pow(bessel_arg,2)/16+torch.pow(bessel_arg,4)/384
    
    p = 1j*Constants.k*distance
    phase = torch.e**(p)
    
    trans_matrix=2*Constants.P_ref*torch.multiply(torch.divide(phase,distance),directivity)
    return trans_matrix

def forward_model_batched(points, transducers = TRANSDUCERS):

    '''
    @private
    computed batched piston model for acoustic wave propagation
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns forward propagation matrix \\
    '''
    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]
    
    # p = torch.permute(points,(0,2,1))
    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    distance_axis = (transducers - points) **2
    distance = torch.sqrt(torch.sum(distance_axis,dim=2))
    planar_distance= torch.sqrt(torch.sum(distance_axis[:,:,0:2,:],dim=2))
    
    bessel_arg=Constants.k*Constants.radius*torch.divide(planar_distance,distance)
    directivity=1/2-torch.pow(bessel_arg,2)/16+torch.pow(bessel_arg,4)/384
    
    p = 1j*Constants.k*distance
    phase = torch.e**(p)

    trans_matrix=2*Constants.P_ref*torch.multiply(torch.divide(phase,distance),directivity)

    return trans_matrix.permute((0,2,1))

def green_propagator(points:Tensor, board:Tensor, k:float=Constants.k) -> Tensor:
    '''
    Computes the Green's function propagation matrix from `board` to `points` \n
    :param points: Points to use
    :param board: transducers to use
    :param k: Wavenumber of sound to use
    :return: Green propagation matrix
    '''

    B = points.shape[0]
    N = points.shape[2]
    M = board.shape[0]
    board = board.unsqueeze(0).unsqueeze_(3)
    points = points.unsqueeze(1)
    
    # distances_axis = torch.abs(points-board)
    distances_axis = (points-board)**2
    distances = torch.sqrt(torch.sum(distances_axis, dim=2))


    
    # green = -1* (torch.exp(1j*k*distances)) / (4 * Constants.pi *distances)
    green = -1* (torch.exp(1j*k*distances)) / (4 * Constants.pi *distances)


    return green.mT