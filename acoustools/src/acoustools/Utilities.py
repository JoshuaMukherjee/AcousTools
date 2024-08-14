import torch, math, sys
import acoustools.Constants as Constants

torch.cuda.empty_cache()

from typing import Literal
from types import FunctionType
from torch import Tensor

DTYPE = torch.complex64
'''
Data type to use for matricies - use `.to(DTYPE)` to convert
'''

device:Literal['cuda','cpu'] = 'cuda' if torch.cuda.is_available() else 'cpu' 
'''Constant storing device to use, `cuda` if cuda is available else cpu. \n
Use -cpu when running python to force cpu use'''
device = device if '-cpu' not in sys.argv else 'cpu'


def create_board(N:int, z:float) -> Tensor: 
    '''
    Create a single transducer array \n
    :param N: Number of transducers + 1 per side eg for 16 transducers `N=17`
    :param z: z-coordinate of board
    :return: tensor of transducer positions
    '''
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1)).to(device)
    x, y = torch.meshgrid(grid_vec,grid_vec,indexing="ij")
    x = x.to(device)
    y= y.to(device)
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_y=torch.reshape(y,(torch.numel(y),1))
    trans_z=z*torch.ones((torch.numel(x),1)).to(device)
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos
  
def transducers() -> Tensor:
  '''
  :return: the 'standard' transducer arrays with 2 16x16 boards at `z = +-234/2 `
  '''
  return torch.cat((create_board(17,BOARD_POSITIONS),create_board(17,-1*BOARD_POSITIONS)),axis=0).to(device)

# BOARD_POSITIONS = .234/2
BOARD_POSITIONS:float = 0.2365/2
'''
Static variable for the z-position of the boards, positive for top board, negative for bottom board
'''

TRANSDUCERS:Tensor = transducers()
'''
Static variable for `transducers()` result
'''
TOP_BOARD:Tensor = create_board(17,BOARD_POSITIONS)
'''
Static variable for a 16x16 array at `z=.234/2` - top board of a 2 array setup
'''
BOTTOM_BOARD:Tensor = create_board(17,-1*BOARD_POSITIONS)
'''
Static variable for a 16x16 array at `z=-.234/2` - bottom board of a 2 array setup
'''

def is_batched_points(points:Tensor) -> bool:
    '''
    :param points: `Tensor` of points
    :return: `True` is points has a batched shape
    '''
    if len(points.shape)> 2 :
        return True
    else:
        return False

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

def compute_gradients(points, transducers = TRANSDUCERS):
    '''
    @private
    Computes the components to be used in the analytical gradient of the piston model, shouldnt be useed use `forward_model_grad` to get the gradient \\
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns (F,G,H, partialFpartialX, partialGpartialX, partialHpartialX, partialFpartialU, partialUpartiala)
    '''
    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]

    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    diff = transducers - points
    distances = torch.sqrt(torch.sum(diff**2, 2))
    planar_distance= torch.sqrt(torch.sum((diff**2)[:,:,0:2,:],dim=2))
    

    #Partial derivates of bessel function section wrt xyz
    sin_theta = torch.divide(planar_distance,distances) 
    partialFpartialU = -1* (Constants.k**2 * Constants.radius**2)/4 * sin_theta + (Constants.k**4 * Constants.radius**4)/48 * sin_theta**3
    partialUpartiala = torch.ones_like(diff)
    
    diff_z = torch.unsqueeze(diff[:,:,2,:],2)
    diff_z = diff_z.expand((-1,-1,2,-1))
    
    denom = torch.unsqueeze((planar_distance*distances**3),2)
    denom = denom.expand((-1,-1,2,-1))
    
    partialUpartiala[:,:,0:2,:] = -1 * (diff[:,:,0:2,:] * diff_z**2) / denom
    partialUpartiala[:,:,2,:] = (diff[:,:,2,:] * planar_distance) / distances**3

    partialFpartialU = torch.unsqueeze(partialFpartialU,2)
    partialFpartialU = partialFpartialU.expand((-1,-1,3,-1))
    partialFpartialX  = partialFpartialU * partialUpartiala

    #Grad of Pref / d(xt,t)
    dist_expand = torch.unsqueeze(distances,2)
    dist_expand = dist_expand.expand((-1,-1,3,-1))
    partialGpartialX = (Constants.P_ref * diff) / dist_expand**3

    #Grad of e^ikd(xt,t)
    partialHpartialX = 1j * Constants.k * (diff / dist_expand) * torch.e**(1j * Constants.k * dist_expand)

    #Combine
    bessel_arg=Constants.k*Constants.radius*torch.divide(planar_distance,distances)
    F=1-torch.pow(bessel_arg,2)/8+torch.pow(bessel_arg,4)/192
    F = torch.unsqueeze(F,2)
    F = F.expand((-1,-1,3,-1))

    G = Constants.P_ref / dist_expand
    H = torch.e**(1j * Constants.k * dist_expand)

    return F,G,H, partialFpartialX, partialGpartialX, partialHpartialX, partialFpartialU, partialUpartiala

def forward_model_grad(points:Tensor, transducers:Tensor|None = None) -> tuple[Tensor]:
    '''
    Computes the analytical gradient of the piston model\n
    :param points: Point position to compute propagation to 
    :param transducers: The Transducer array, default two 16x16 arrays 
    :return: derivative of forward model wrt x,y,z position

    ```Python
    from acoustools.Utilities import forward_model_grad

    Fx, Fy, Fz = forward_model_grad(points,transducers=board)
    Px  = torch.abs(Fx@activations) #gradient wrt x position
    Py  = torch.abs(Fy@activations) #gradient wrt y position
    Pz  = torch.abs(Fz@activations) #gradient wrt z position

    ```
    '''
    if transducers is None:
        transducers=TRANSDUCERS

    F,G,H, partialFpartialX, partialGpartialX, partialHpartialX,_,_ = compute_gradients(points, transducers)
    derivative = G*(H*partialFpartialX + F*partialHpartialX) + F*H*partialGpartialX

    return derivative[:,:,0,:].permute((0,2,1)), derivative[:,:,1,:].permute((0,2,1)), derivative[:,:,2,:].permute((0,2,1))

def forward_model_second_derivative_unmixed(points:Tensor, transducers:Tensor|None = None) ->Tensor:
    '''
    Computes the second degree unmixed analytical gradient of the piston model\n
    :param points: Point position to compute propagation to 
    :param transducers: The Transducer array, default two 16x16 arrays 
    :return: second degree unmixed derivatives of forward model wrt x,y,z position Pxx, Pyy, Pzz
    '''

    if transducers is None:
        transducers= TRANSDUCERS

    F,G,H, partialFpartialX, partialGpartialX, partialHpartialX , partialFpartialU, partialUpartialX = compute_gradients(points, transducers)

    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]

    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    diff = transducers - points
    distance_axis = diff**2
    distances = torch.sqrt(torch.sum(distance_axis, 2))
    planar_distance= torch.sqrt(torch.sum(distance_axis[:,:,0:2,:],dim=2))

    partial2fpartialX2 = torch.ones_like(diff)

    dx = distance_axis[:,:,0,:]
    dy = distance_axis[:,:,1,:]
    dz = distance_axis[:,:,2,:]
    
    planar_square = planar_distance**2
    distances_square  = distances**2

    partial2fpartialX2[:,:,0,:] = (-2*dx**2*planar_square*distances_square + dy**2 * distances_square + planar_square * (2*dx**2 - dy**2 - dz**2)) / (planar_square**(3/4) * distances_square**(5/2))
    partial2fpartialX2[:,:,1,:] = (planar_square**2 * (-1*(dx*2-2*dy**2 + dz**2)) -2*dy**2*planar_square*distances_square +dx**2*distances_square**2) / (planar_square**(3/4) * distances_square**(5/2))
    partial2fpartialX2[:,:,2,:] = planar_distance * (((3*dz**2)/(distances_square**(5/2))) - (1/(distances_square**(3/2)))) #This could be wrong?

    sin_theta = torch.divide(planar_distance,distances)
    partial2Fpartialf2 = -1 * (Constants.k**2 * Constants.radius**2)/4 + (Constants.k**4 * Constants.radius**4)/16 * sin_theta**2

    partial2Fpartialf2 = torch.unsqueeze(partial2Fpartialf2,2)
    partial2Fpartialf2 = partial2Fpartialf2.expand((-1,-1,3,-1))
    partial2FpartialX2 = partialUpartialX**2 * partial2Fpartialf2 + partial2fpartialX2*partialFpartialU

    dist_expand = torch.unsqueeze(distances,2)
    dist_expand = dist_expand.expand((-1,-1,3,-1))

    partialdpartialX =  diff / dist_expand

    partial2HpartialX2 = Constants.k * torch.e**(1j*Constants.k*dist_expand) * (dist_expand * (Constants.k * diff*partialdpartialX + 1j)+1j*diff*partialdpartialX) / dist_expand**2

    partial2GpartialX2 = (Constants.P_ref * (3*diff * partialdpartialX + dist_expand)) / (dist_expand**4)

    derivative = 2*partialHpartialX * (G * partialFpartialX + F * partialGpartialX) + H*(G*partial2FpartialX2 + 2*partialFpartialX*partialGpartialX + F*partial2GpartialX2) + F*G*partial2HpartialX2
    
    return derivative[:,:,0,:].permute((0,2,1)), derivative[:,:,1,:].permute((0,2,1)), derivative[:,:,2,:].permute((0,2,1))

def forward_model_second_derivative_mixed(points: Tensor, transducers:Tensor|None = None)->Tensor:
    '''
    Computes the second degree mixed analytical gradient of the piston model\n
    :param points: Point position to compute propagation to 
    :param transducers: The Transducer array, default two 16x16 arrays 
    Returns second degree mixed derivatives of forward model wrt x,y,z position - Pxy, Pxz, Pyz
    '''

    if transducers is None:
        transducers= TRANSDUCERS
    
    F,G,H, Fa, Ga, Ha , Fu, Ua = compute_gradients(points, transducers)

    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]

    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))
    
    diff = transducers - points
    distance_axis = diff**2
    distances = torch.sqrt(torch.sum(distance_axis, 2))
    planar_distance= torch.sqrt(torch.sum(distance_axis[:,:,0:2,:],dim=2))

    sin_theta = torch.divide(planar_distance,distances)

    dx = distance_axis[:,:,0,:] #Should this be distance_axis or diff
    dy = distance_axis[:,:,1,:]
    dz = distance_axis[:,:,2,:]

    # distances_sqaured = distances**2
    distances_five = distances**5
    distances_cube = distances**3
    # planar_distance_squared = planar_distance**2

    Fxy = torch.ones((B,M,1,N))
    Fxz = torch.ones((B,M,1,N))
    Fyz = torch.ones((B,M,1,N))
    
    planar_distance_distances_five = planar_distance * distances_five
    Uxy = -1*(dx*dy*dz**2 * (4*dx**2+4*dy**2+dz**2)) / (planar_distance**3 * distances_five)
    Uxz = ((dx*dz) * (2*dx**2 + 2*dy**2 -dz**2)) / planar_distance_distances_five
    Uyz = ((dy*dz)*(2*dx**2 + 2*dy**2 - dz**2)) / planar_distance_distances_five

    Ux = Ua[:,:,0,:]
    Uy = Ua[:,:,1,:]
    Uz = Ua[:,:,2,:]

    # F_second_U = -1 * (Constants.k**2 * Constants.radius**2)/4 + (Constants.k**4 * Constants.radius**4)/16 * sin_theta**2
    # F_first_U = -1* (Constants.k**2 * Constants.radius**2)/4 * sin_theta + (Constants.k**4 * Constants.radius**4)/48 * sin_theta**3

    F_second_U = -1/8 *Constants.k**3*Constants.radius**3  + 1/32 * Constants.k**5 * Constants.radius**5 * sin_theta**2 - 5/3072 * sin_theta**4 * Constants.k**7 * Constants.radius**7
    F_first_U = -1* (Constants.k**3 * Constants.radius**3)/8 * sin_theta + 1/96 * Constants.k**5 * Constants.radius**5 * sin_theta**3 - 1/3072 * sin_theta**7 * Constants.k**7 * Constants.radius**7

    Fxy = Ux * Uy * F_second_U + Uxy*F_first_U
    Fxz = Ux * Uz * F_second_U + Uxz*F_first_U
    Fyz = Uy * Uz * F_second_U + Uyz*F_first_U

    dist_xy = (dx*dy) / distances_cube
    dist_xz = (dx*dz) / distances_cube
    dist_yz = (dy*dz) / distances_cube

    dist_x = -1 * dx / distances
    dist_y = -1 * dy / distances
    dist_z = -1 * dz / distances

    Hxy = -Constants.k *  H[:,:,0,:] * (Constants.k * dist_y * dist_x - 1j*dist_xy)
    Hxz = -Constants.k *  H[:,:,1,:] * (Constants.k * dist_z * dist_x - 1j*dist_xz)
    Hyz = -Constants.k *  H[:,:,2,:] * (Constants.k * dist_y * dist_z - 1j*dist_yz)

    Gxy = Constants.P_ref * (2*dist_y*dist_x - distances * dist_xy) / distances_cube
    Gxz = Constants.P_ref * (2*dist_z*dist_x - distances * dist_xz) / distances_cube
    Gyz = Constants.P_ref * (2*dist_z*dist_y - distances * dist_yz) / distances_cube

    Fx = Fa[:,:,0,:] 
    Fy = Fa[:,:,1,:] 
    Fz = Fa[:,:,2,:]

    Hx = Ha[:,:,0,:] 
    Hy = Ha[:,:,1,:] 
    Hz = Ha[:,:,2,:]

    Gx = Ga[:,:,0,:] 
    Gy = Ga[:,:,1,:] 
    Gz = Ga[:,:,2,:] 

    F_ = F[:,:,0,:]
    H_ = H[:,:,0,:]
    G_ = G[:,:,0,:]

    Pxy =  (H_*(Fx * Gy + Fy*Gx + Fxy*G_ + F_*Gxy) + G_ * (Fx*Hy+Fy*Hx + F_*Hxy) + F_*(Gx*Hy + Gy*Hx))
    Pxz =  (H_*(Fx * Gz + Fz*Gx + Fxz*G_ + F_*Gxz) + G_ * (Fx*Hz+Fz*Hx + F_*Hxz) + F_*(Gx*Hz + Gz*Hx))
    Pyz =  (H_*(Fy * Gz + Fz*Gy + Fyz*G_ + F_*Gyz) + G_ * (Fy*Hz+Fz*Hy + F_*Hyz) + F_*(Gy*Hz + Gz*Hy))


    return Pxy.permute(0,2,1),Pxz.permute(0,2,1), Pyz.permute(0,2,1)
    
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

def permute_points(points: Tensor,index: int,axis:int=0) -> Tensor:
    '''
    Permutes axis of a tensor \n
    :param points: Tensor to permute
    :param index: Indexes describing order to perumte to 
    :param axis: Axis to permute. Default `0`
    :return: permuted points
    '''
    if axis == 0:
        return points[index,:,:,:]
    if axis == 1:
        return points[:,index,:,:]
    if axis == 2:
        return points[:,:,index,:]
    if axis == 3:
        return points[:,:,:,index]

def swap_output_to_activations(out_mat,points):
    '''
    @private
    '''
    acts = None
    for i,out in enumerate(out_mat):
        out = out.T.contiguous()
        pressures =  torch.view_as_complex(out)
        A = forward_model(points[i]).to(device)
        if acts == None:
            acts =  A.T @ pressures
        else:
            acts = torch.stack((acts,A.T @ pressures),0)
    return acts

def convert_to_complex(matrix: Tensor) -> Tensor:
    '''
    Comverts a real tensor of shape `B x M x N` to a complex tensor of shape `B x M/2 x N` 
    :param matrix: Matrix to convert
    :return: converted complex tensor
    '''
    # B x 1024 x N (real) -> B x N x 512 x 2 -> B x 512 x N (complex)
    matrix = torch.permute(matrix,(0,2,1))
    matrix = matrix.view((matrix.shape[0],matrix.shape[1],-1,2))
    matrix = torch.view_as_complex(matrix.contiguous())
    return torch.permute(matrix,(0,2,1))

def get_convert_indexes(n:int=512) -> Tensor:
    '''
    Gets indexes to swap between transducer order for acoustools and OpenMPD for two boards\n
    Use: `row = row[:,FLIP_INDEXES]` and invert with `_,INVIDX = torch.sort(IDX)` 
    :param n: number of Transducers
    :return: Indexes
    '''

    indexes = torch.arange(0,n)
    # Flip top board
    indexes[:256] = torch.flip(indexes[:256],dims=[0])
    if n > 256:
        indexes[256:] = torch.flatten(torch.flip(torch.reshape(indexes[256:],(16,-1)),dims=[1]))
    return indexes



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
        points[:,0,:] = x
    
    if y is not None:
        points[:,1,:] = y
    
    if z is not None:
        points[:,2,:] = z

    return points
    
def add_lev_sig(activation:Tensor, board:Tensor|None=None, 
                mode:Literal['Focal', 'Trap', 'Vortex','Twin']='Trap', sig:Tensor|None=None, return_sig:bool=False) -> Tensor:
    '''
    Adds signature to hologram for a board \n
    :param activation: Hologram input
    :param board: Board to use
    :param mode: Type of signature to add, should be one of
    * Focal: No signature
    * Trap: Add $\\pi$ to the top board - creates a trap
    * Vortex: Add a circular signature to create a circular trap
    * Twin: Add $\\pi$ to half of the board laterally to create a twin trap
    :param sig: signature to add to top board. If `None` then value is determined by value of `mode`
    :return: hologram with signature added

    ```Python
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs

    p = create_points(1,x=0,y=0,z=0)
    x = wgs(p, board=board)
    x_sig, sig = add_lev_sig(x.clone(), mode=mode, return_sig=True, board=board)

    ```
    '''
    if board is None:
        board = TRANSDUCERS

    act = activation.clone().to(device)

    s = act.shape
    B = s[0]

    act = torch.reshape(act,(B,-1, 256))

    # act[:,0,:] = torch.e**(1j*(sig + torch.angle(act[:,0,:].clone())))
    if sig is None:
        sig = torch.ones_like(act)
        if mode == 'Trap':
            sig = torch.cat([torch.ones_like(act[:,0,:]) * torch.pi, torch.zeros_like(act[:,0,:])])
        if mode == 'Focal':
            sig = torch.zeros_like(act)
        if mode == 'Vortex':
            plane = board[:,0:2]
            sig = torch.atan2(plane[:,0], plane[:,1]).unsqueeze(0).unsqueeze(2).reshape((B,-1, 256))
        if mode == 'Twin':
            plane = board[:,0:2]
            sig = torch.zeros_like(sig) + torch.pi * (plane[:,0] > 0).unsqueeze(0).unsqueeze(2).reshape((B,-1, 256))


    x = torch.abs(act) * torch.exp(1j* (torch.angle(act) + sig))

    x = torch.reshape(x,s)

    if return_sig:
        return x, sig
    return x

def generate_gorkov_targets(N:int,B:int=1, max_val:float=0, min_val:float=-1e-4) -> Tensor:
    '''
    Generates a tensor of random Gor'kov potential values\n
    If `B=0` will return tensor with shape of `Nx1` else  will have shape `BxNx1`\n
    :param N: Number of values per batch
    :param B: Number of batches to produce
    :param max_val: Maximum value that can be generated. Default: `0`
    :param min_val: Minimum value that can be generated. Default: `-1e-4`
    :return: tensor of values
    '''
    if B > 0:
        targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    else:
         targets = torch.FloatTensor(N,1).uniform_(min_val,max_val).to(device)
    return targets

def generate_pressure_targets(N:int,B:int=1, max_val:float=5000, min_val:float=3000) -> Tensor:
    '''
    Generates a tensor of random pressure values\\
    :param N: Number of values per batch
    :param B: Number of batches to produce
    :param max_val: Maximum value that can be generated. Default: `5000`
    :param min_val: Minimum value that can be generated. Default: `3000`
    Returns tensor of values
    '''
    targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    return targets

def return_matrix(x,y,mat=None):
    '''
    @private
    Returns value of parameter `mat` - For compatibility with other functions
    '''
    return mat

def write_to_file(activations:Tensor,fname:str,num_frames:int, num_transducers:int=512, flip:bool=True) -> None:
    '''
    Writes each hologram in `activations` to the csv `fname` in order expected by OpenMPD \n
    :param activations: List of holograms
    :param fname: Name of file to write to, expected to end in `.csv`
    :param num_frames: Number of frames in `activations` 
    :param num_transducers: Number of transducers in the boards used. Default:512
    :param flip: If True uses `get_convert_indexes` to swap order of transducers to be the same as OpenMPD expects. Default: `True`
    '''
    output_f = open(fname,"w")
    output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    
    for row in activations:
        row = torch.angle(row).squeeze_()
        
        if flip:
            FLIP_INDEXES = get_convert_indexes()
            row = row[FLIP_INDEXES]
            

       
        for i,phase in enumerate(row):
                    output_f.write(str(phase.item()))
                    if i < num_transducers-1:
                        output_f.write(",")
                    else:
                        output_f.write("\n")

    output_f.close()

def get_rows_in(a_centres, b_centres, expand = True):
    '''
    @private
    Takes two tensors and returns a mask for `a_centres` where a value of true means that row exists in `b_centres` \\
    Asssumes in form 1x3xN -> returns mask over dim 1\\
    `a_centres` Tensor of points to check for inclusion in `b_centres` \\
    `b_centres` Tensor of points which may or maynot contain some number of points in `a_centres`\\
    `expand` if True returns mask as `1x3xN` if False returns mask as `1xN`. Default: True\\
    Returns mask for all rows in `a_centres` which are in `b_centres`
    '''

    M = a_centres.shape[2] #Number of total elements
    R = b_centres.shape[2] #Number of elements in b

    a_reshape = torch.unsqueeze(a_centres,3).expand(-1, -1, -1, R)
    b_reshape = torch.unsqueeze(b_centres,2).expand(-1, -1, M, -1)

    mask = b_reshape == a_reshape
    mask = mask.all(dim=1).any(dim=2)

    if expand:
        return mask.unsqueeze(1).expand(-1,3,-1)
    else:
        return mask

def read_phases_from_file(file: str, invert:bool=True, top_board:bool=False, ignore_first_line:bool=True):
    '''
    Gets phases from a csv file, expects a csv with each row being one geometry
    :param file: The file path to read from
    :param invert: Convert transducer order from OpenMPD -> Acoustools order. Default True
    :param top_board: if True assumes only the top board. Default False
    :param ignore_first_line: If true assumes header is the first line
    :return: phases
    '''
    phases_out = []
    line_one = True
    with open(file, "r") as f:
        for line in f.readlines():
            if ignore_first_line and line_one:
                line_one = False
                continue
            phases = line.rstrip().split(",")
            phases = [float(p) for p in phases]
            phases = torch.tensor(phases).to(device).unsqueeze_(1)
            phases = torch.exp(1j*phases)
            if invert:
                if not top_board:
                    IDX = get_convert_indexes()
                    _,INVIDX = torch.sort(IDX)
                    phases = phases[INVIDX]
                else:
                    for i in range(16):
                    #    print(torch.flipud(TOP_BOARD[i*16:(i+1)*16]))
                       phases[i*16:(i+1)*16] = torch.flipud(phases[i*16:(i+1)*16])
            phases_out.append(phases)
    phases_out = torch.stack(phases_out)
    return phases_out
            
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