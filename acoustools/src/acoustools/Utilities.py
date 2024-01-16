import torch, math, sys
import acoustools.Constants as Constants


device = 'cuda' if torch.cuda.is_available() else 'cpu' 
'''Constant storing device to use, `cuda` if cuda is available else cpu. \\ 
Use -cpu when running python to force cpu use'''
device = device if '-cpu' not in sys.argv else 'cpu'


def create_board(N, z): 
    '''
    Create a single transducer array \\
    `N` Number of transducers + 1 per side eg for 16 transducers `N=17`\\
    `z` z-coordinate of board\\
    Returns tensor of transducer positions\\
    Written by Giorgos Christopoulos, 2022
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
  
def transducers():
  '''
  Returns the 'standard' transducer arrays with 2 16x16 boards at `z = +-234/2 `\\
  Written by Giorgos Christopoulos, 2022
  '''
  return torch.cat((create_board(17,.234/2),create_board(17,-.234/2)),axis=0).to(device)

TRANSDUCERS = transducers()
'''
Static variable for `transducers()` result
'''
TOP_BOARD = create_board(17,.234/2)
'''
Static variable for a 16x16 array at `z=.234/2` - top board of a 2 array setup
'''
BOTTOM_BOARD = create_board(17,-.234/2)
'''
Static variable for a 16x16 array at `z=-.234/2` - bottom board of a 2 array setup
'''

def forward_model(points, transducers = TRANSDUCERS):
    '''
    Compute the piston model for acoustic wave propagation NOTE: Unbatched, use `forward_model_batched` for batched computation \\
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns forward propagation matrix \\
    Written by Giorgos Christopoulos, 2022
    '''
    
    m=points.size()[1]
    n=transducers.size()[0]
    k=2*math.pi/0.00865
    radius=0.005
    
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

    bessel_arg=k*radius*torch.divide(planar_distance,distance) #planar_dist / dist = sin(theta)

    directivity=1/2-torch.pow(bessel_arg,2)/16+torch.pow(bessel_arg,4)/384
    
    p = 1j*k*distance
    phase = torch.e**(p)
    
    trans_matrix=2*Constants.P_ref*torch.multiply(torch.divide(phase,distance),directivity)
    return trans_matrix

def compute_gradients(points, transducers = TRANSDUCERS):
    '''
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

def forward_model_grad(points, transducers = TRANSDUCERS):
    '''
    Computes the analytical gradient of the piston model\\
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns derivative of forward model wrt x,y,z position
    '''
    F,G,H, partialFpartialX, partialGpartialX, partialHpartialX,_,_ = compute_gradients(points, transducers)
    derivative = G*(H*partialFpartialX + F*partialHpartialX) + F*H*partialGpartialX

    return derivative[:,:,0,:].permute((0,2,1)), derivative[:,:,1,:].permute((0,2,1)), derivative[:,:,2,:].permute((0,2,1))

def forward_model_second_derivative_unmixed(points, transducers = TRANSDUCERS):
    '''
    Computes the second degree unmixed analytical gradient of the piston model\\
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns second degree unmixed derivatives of forward model wrt x,y,z position Pxx, Pyy, Pzz
    '''
    F,G,H, partialFpartialX, partialGpartialX, partialHpartialX , partialFpartialU, partialUpartialX= compute_gradients(points, transducers)

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
    partial2fpartialX2[:,:,2,:] = planar_distance * (((3*dz**2)/distances_square**(5/2)) - (1/distances_square**(3/2)))

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

def forward_model_second_derivative_mixed(points, transducers = TRANSDUCERS):
    '''
    Computes the second degree mixed analytical gradient of the piston model\\
    `points` Point position to compute propagation to \\
    `transducers` The Transducer array, default two 16x16 arrays \\
    Returns second degree mixed derivatives of forward model wrt x,y,z position - Pxy, Pxz, Pyz
    '''
    
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

    dx = distance_axis[:,:,0,:]
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

    F_second_U = -1 * (Constants.k**2 * Constants.radius**2)/4 + (Constants.k**4 * Constants.radius**4)/16 * sin_theta**2
    F_first_U = -1* (Constants.k**2 * Constants.radius**2)/4 * sin_theta + (Constants.k**4 * Constants.radius**4)/48 * sin_theta**3

    F_ab_term_1 = Ux * Uy * F_second_U
    Fxy = F_ab_term_1 + Uxy*F_first_U
    Fxz = F_ab_term_1 + Uxz*F_first_U
    Fyz = F_ab_term_1 + Uyz*F_first_U

    dist_xy = (dx*dy) / distances_cube
    dist_xz = (dz*dz) / distances_cube
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

    Pxy = H_*(Fx * Gy + Fy*Gx + Fxy*G_ + F_*Gxy) + G_ * (Fx*Hy+Fy*Hx + F_*Hxy) + F_*(Gx*Hy + Gy*Hx)
    Pxz = H_*(Fx * Gz + Fz*Gx + Fxz*G_ + F_*Gxz) + G_ * (Fx*Hz+Fz*Hx + F_*Hxz) + F_*(Gx*Hz + Gz*Hx)
    Pyz = H_*(Fy * Gz + Fz*Gy + Fyz*G_ + F_*Gyz) + G_ * (Fy*Hz+Fz*Hy + F_*Hyz) + F_*(Gy*Hz + Gz*Hy)

    return Pxy.permute(0,2,1), Pxz.permute(0,2,1), Pyz.permute(0,2,1)

def forward_model_batched(points, transducers = TRANSDUCERS):
    '''
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
    
def propagate(activations, points,board=TRANSDUCERS, A=None):
    '''
    Propagates a hologram to target points\\
    `activations` Hologram to use\\
    `points` Points to propagate to\\
    `board` The Transducer array, default two 16x16 arrays\\
    `A` The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`\\
    Returns point activations
    '''
    if A is None:
        A = forward_model_batched(points,board).to(device)
    prop = A@activations
    prop = torch.squeeze(prop, 2)
    return prop

def propagate_abs(activations, points,board=TRANSDUCERS, A=None):
    '''
    Propagates a hologram to target points and returns pressure - Same as `torch.abs(propagate(activations, points,board, A))`\\
    `activations` Hologram to use\\
    `points` Points to propagate to\\
    `board` The Transducer array, default two 16x16 arrays\\
    `A` The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`\\
    Returns point pressure
    '''
    out = propagate(activations, points,board,A=A)
    return torch.abs(out)

def permute_points(points,index,axis=0):
    '''
    Permutes axis a tensor\\
    'points' Tensor to permute\\
    'index' Indexes describing order to perumte to \\
    'axis' Axis to permute. Default `0`\\
    Returns permuted points
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

def convert_to_complex(matrix):
    '''
    Comverts a real tensor of shape `B x M x N` to a complex tensor of shape `B x M/2 x N` \\
    `matrix` Matrix to convert\\
    Returns converted complex tensor
    '''
    # B x 1024 x N (real) -> B x N x 512 x 2 -> B x 512 x N (complex)
    matrix = torch.permute(matrix,(0,2,1))
    matrix = matrix.view((matrix.shape[0],matrix.shape[1],-1,2))
    matrix = torch.view_as_complex(matrix.contiguous())
    return torch.permute(matrix,(0,2,1))

def get_convert_indexes():
    '''
    Gets indexes to swap between transducer order for acoustools and OpenMPD\\
    Use: `row = row[FLIP_INDEXES]` and invert with `_,INVIDX = torch.sort(IDX)` \\
    Returns Indexes
    '''
    
    board = transducers()
    board[512//2:,0] = torch.flipud(board[512//2:,0])
    board[:,1] = torch.flipud(board[:,1])
    board[:,2] = torch.flipud(board[:,2])
    indexes = []

    for t,row in enumerate(board):
        for b,row_b in enumerate(transducers()):
            if torch.all(row == row_b):
                indexes.append(b)
    indexes = torch.as_tensor(indexes)


    return indexes

def create_points(N,B=1,x=None,y=None,z=None, min_pos=-0.06, max_pos = 0.06):
    '''
    Creates a random set of N points in B batches in shape `Bx3xN`\\
    `N` Number of points per batch\\
    `B` Number of Batches\\
    `x` if not None all points will have this as their x position. Default: `None`\\
    `y` if not None all points will have this as their y position. Default: `None`\\
    `z` if not None all points will have this as their z position. Default: `None`\\
    '''
    points = torch.FloatTensor(B, 3, N).uniform_(min_pos,max_pos).to(device)
    if x is not None:
        points[:,0,:] = x
    
    if y is not None:
        points[:,1,:] = y
    
    if z is not None:
        points[:,2,:] = z

    return points
    
def add_lev_sig(activation, sig=torch.pi):
    '''
    Adds pi to the top board of a 2x16x16 board setup - converts focal points to traps\\
    `activation` Hologram input for 2x16x16 board\\
    `sig` signature to add to top board. Default: `pi`\\
    Returns hologram with signature added
    '''
    act = activation.clone().to(device)

    s = act.shape
    B = s[0]

    act = torch.reshape(act,(B,2, 256))

    act[:,0,:] = torch.e**(1j*(sig + torch.angle(act[:,0,:].clone())))
    act = torch.reshape(act,s)

    return act

def generate_gorkov_targets(N,B=1, max_val=0, min_val=-1e-4):
    '''
    Generates a tensor of random Gor'kov potential values\\
    If `B=0` will return tensor with shape of `Nx1` else  will have shape `BxNx1`\\
    `N` Number of values per batch\\
    `B` Number of batches to produce\\
    `max_val` Maximum value that can be generated. Default: `0`\\
    `max_val` Minimum value that can be generated. Default: `-1e-4`\\
    Returns tensor of values
    '''
    if B > 0:
        targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    else:
         targets = torch.FloatTensor(N,1).uniform_(min_val,max_val).to(device)
    return targets

def generate_pressure_targets(N,B=1, max_val=10000, min_val=7000):
    '''
    Generates a tensor of random pressure values\\
    `N` Number of values per batch\\
    `B` Number of batches to produce\\
    `max_val` Maximum value that can be generated. Default: `10000`\\
    `max_val` Minimum value that can be generated. Default: `7000`\\
    Returns tensor of values
    '''
    targets = torch.FloatTensor(B, N,1).uniform_(min_val,max_val).to(device)
    return targets

def return_matrix(x,y,mat=None):
    '''
    Returns value of parameter `mat` - For compatibility with other functions
    '''
    return mat

def write_to_file(activations,fname,num_frames, num_transducers=512, flip=True):
    '''
    Writes each hologram in `activations` to the csv `fname` in order expected by OpenMPD\\
    `activations` List of holograms\\
    `fname` Name of file ro write to, expected to end in `.csv`\\
    `num_frames` Number of frames in `activations` \\
    `num_transducers` Number of transducers in the boards used. Default:512\\
    `flip`: If True uses `get_convert_indexes` to swap order of transducers to be the same as OpenMPD expects. Default: `True`\\ 
    '''
    
    row = torch.angle(activations).squeeze_()
    
    if flip:
        FLIP_INDEXES = get_convert_indexes()
        row = row[FLIP_INDEXES]
        

    output_f = open(fname,"w")
    output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    for i,phase in enumerate(row):
                output_f.write(str(phase.item()))
                if i < num_transducers-1:
                    output_f.write(",")
                else:
                    output_f.write("\n")

    output_f.close()

def get_rows_in(a_centres, b_centres, expand = True):
    '''
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



if __name__ == "__main__":
    from acoustools.Solvers import wgs,wgs_batch
    from acoustools.Gorkov import gorkov_fin_diff, gorkov_analytical

    
    points = create_points(4,2)
    

    
    F = forward_model_batched(points)
    _, _, x = wgs_batch(F,torch.ones(4,1).to(device)+0j,200)

    Fx, Fy, Fz = forward_model_grad(points)
    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points)
    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points)

    x = add_lev_sig(x)
    p = torch.abs(F@x)
    Px = torch.abs(Fx@x)
    Py = torch.abs(Fy@x)
    Pz = torch.abs(Fz@x)
    Pxx = torch.abs(Fxx@x)
    Pyy = torch.abs(Fyy@x)
    Pzz = torch.abs(Fzz@x)
    Pxy = torch.abs(Fxy@x)
    Pxz = torch.abs(Fxz@x)
    Pyz = torch.abs(Fyz@x)

    print(Constants.V,Constants.p_0,Constants.c_0 )
    
    K1 = Constants.V / (4*Constants.p_0*Constants.c_0**2)
    K2 = 3*Constants.V / (4*(2*Constants.f**2 * Constants.p_0))

    # print(K1, K2)

    print(Px, Py, Pz)
    exit()
    single_sum = 2*K2*(Pz+Py+Pz)
    Fx = -1 * (2*p * (K1 * Px - K2*(Pxz+Pxy+Pxx)) - Px*single_sum)
    Fy = -1 * (2*p * (K1 * Py - K2*(Pyz+Pyy+Pxy)) - Py*single_sum)
    Fz = -1 * (2*p * (K1 * Pz - K2*(Pzz+Pyz+Pxz)) - Pz*single_sum)

    # print( Pz, Pzz, Pyz, Pxz)

    # print(2*p * (K1 * Pz - K2*(Pzz+Pyz+Pxz)) , Pz*single_sum)
   
    force = torch.cat([Fx,Fy,Fz],2)
    
    
    print(p)
    print(force)
    

    K1 = Constants.V / (4*Constants.p_0*Constants.c_0**2)
    K2 = 3*Constants.V / (4*(2*Constants.f**2 * Constants.p_0))
    U = K1*p**2 - K2*(Px**2+Py**2+Pz**2)
    print(U)
    print(K1*p**2)
    print((Px**2+Py**2+Pz**2))

    # axis="XYZ"
    # U = gorkov_analytical(x,points,axis=axis)
    # U_fin = gorkov_fin_diff(x, points,axis=axis)
    # print("Gradient function",U.squeeze_())
    # print("Finite differences",U_fin)



    '''

    A1 = forward_model(points[0,:])
    _, _, x1 = wgs(A1,torch.ones(4,1).to(device)+0j,200)

    A2 = forward_model(points[1,:])
    _, _, x2 = wgs(A2,torch.ones(4,1).to(device)+0j,200)

    A = forward_model_batched(points)
    x = torch.stack([x1,x2])
    print(A.shape)
    print(x.shape)
    print(torch.abs(A@x))



    A = forward_model(points[0,:])
    _, _, x = wgs(A,torch.ones(4,1).to(device)+0j,200)
    x = torch.unsqueeze(x,0)
    
    pr = propagate(x,points)
    print(torch.abs(pr))
    
    x2 = add_lev_sig(x)
    pr2 = propagate(x2,points)
    print(torch.abs(pr2))


    from torch.utils.data import DataLoader 
    from Dataset import NaiveDataset
    from Loss_Functions import mse_loss

    net = torch.load("./Models/model_NCNN1_latest.pth")
    # points=  torch.FloatTensor(3,4).uniform_(-.06,.06).to(device)
    dataset = torch.load("./Datasets/NaiveDatasetTrain-4-4.pth")
    data = iter(DataLoader(dataset,1,shuffle=True))

    for p,a,pr,n in data:

        activation_out = do_NCNN(net, p)

        field = propagate(activation_out,p)

        f = torch.unsqueeze(field,0)
        print(mse_loss(torch.abs(f), torch.abs(pr)))
        print(torch.abs(field))

    '''

    '''
    trans_pos(:,2) = flipud(trans_pos(:,2));   
    trans_pos(n/2+1:end,1) = flipud(trans_pos(n/2+1:end,1));
    '''

    # board = transducers()
    # print(board.shape)
    # board[512//2:,0] = torch.flipud(board[512//2:,0]);
    # board[:,1] = torch.flipud(board[:,1]);
    # board[:,2] = torch.flipud(board[:,2]);
    # indexes = []

    # for t,row in enumerate(board):
    #     for b,row_b in enumerate(transducers()):
    #         if torch.all(row == row_b):
    #             indexes.append(b)


    # indexes = torch.as_tensor(indexes)
    # trans = transducers()
    # flipped = trans[indexes]


    # for i,row in enumerate(flipped):
    #     print(row)

    
    
    # print(torch.reshape(board,(16,16,3)))