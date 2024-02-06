import torch

BOARD_POSITIONS = 0.2365/2

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
  return torch.cat((create_board(17,BOARD_POSITIONS),create_board(17,-1*BOARD_POSITIONS)),axis=0).to(device)


pi = 3.1415926535
R = .001 
'''Radius of particle'''
V = 4/3 * pi * R**3
'''Volume of Particle'''
c_0 = 343
'''Speed of sound in air'''
p_0 = 1.2
'''Density of air'''
c_p = 1052
'''Speed of sound in EPS particle'''
p_p = 29.36 
'''density of EPS particle, From Holographic acoustic elements for manipulation of levitated objects'''
f = 40000
'''Frequency of 40KHz sound'''
wavelength = c_0 / f #0.008575
'''Wavelength of 40KHz sound'''
k = 2*pi / wavelength #732.7329804081634
'''Wavenumber of 40KHz sound'''
# radius=0.005 
radius = 0.0045
'''Radius of transducer'''
# P_ref = 8.02 #old value
P_ref = 0.17*20 #3.4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANSDUCERS = transducers()

def ph_thresh(z_last,z,threshold):
    '''
    Phase threshhold between two timesteps point phases, clamps phase changes above `threshold` to be `threshold`\\
    `z_last` point activation at timestep t-1\\
    `z` point activation at timestep t\\
    `threshold` maximum allowed phase change\\
    returns constrained point activations
    '''

    ph1 = torch.angle(z_last)
    ph2 = torch.angle(z)
    dph = ph2 - ph1
    
    dph = torch.atan2(torch.sin(dph),torch.cos(dph)) 
    
    dph[dph>threshold] = threshold
    dph[dph<-1*threshold] = -1*threshold
    
    ph2 = ph1 + dph
    z = abs(z)*torch.exp(1j*ph2)
    
    return z

def soft(x,threshold):
    '''
    Soft threshold for a set of phase changes, will return the change - threshold if change > threshold else 0\\
    `x` phase changes\\
    `threshold` Maximum allowed hologram phase change\\
    returns new phase changes
    '''
    y = torch.max(torch.abs(x) - threshold,0).values
    y = y * torch.sign(x)
    return y

def ph_soft(x_last,x,threshold):
    '''
    Soft thresholding for holograms \\
    `x_last` Hologram from timestep t-1\\
    `x` Hologram from timestep t \\
    `threshold` Maximum allowed phase change\\
    returns constrained hologram
    '''
    pi = torch.pi
    ph1 = torch.angle(x_last)
    ph2 = torch.angle(x)
    dph = ph2 - ph1

    dph[dph>pi] = dph[dph>pi] - 2*pi
    dph[dph<-1*pi] = dph[dph<-1*pi] + 2*pi

    dph = soft(dph,threshold)
    ph2 = ph1 + dph
    x = abs(x)*torch.exp(1j*ph2)
    return x

def temporal_wgs(A, y, K,ref_in, ref_out,T_in,T_out):
    '''
    Based off `
    Giorgos Christopoulos, Lei Gao, Diego Martinez Plasencia, Marta Betcke, 
    Ryuji Hirayama, and Sriram Subramanian. 2023. 
    Temporal acoustic point holography.(under submission) (2023)` \\
    WGS solver for hologram where the phase change between frames is constrained\\
    `A` Forward model  to use\\
    `y` initial guess to use normally use `torch.ones(self.N,1).to(device)+0j`\\
    `K` Number of iterations to use\\
    `ref_in` Previous timesteps hologram\\
    `ref_out` Previous timesteps point activations\\
    `T_in` Hologram phase change threshold\\
    `T_out` Point activations phase change threshold\\
    returns (hologram image, point phases, hologram)
    '''
    #ref_out -> points
    #ref_in-> transducers
    AT = torch.conj(A).mT.to(device)
    y0 = y.to(device)
    x = torch.ones(A.shape[2],1).to(device) + 0j
    for kk in range(K):
        z = torch.matmul(A,x)                                   # forward propagate
        z = z/torch.max(torch.abs(z))                           # normalize forward propagated field (useful for next step's division)
        z = ph_thresh(ref_out,z,T_out); 
        
        y = torch.multiply(y0,torch.divide(y,torch.abs(z)))     # update target - current target over normalized field
        y = y/torch.max(torch.abs(y))                           # normalize target
        p = torch.multiply(y,torch.divide(z,torch.abs(z)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram    
        x = ph_thresh(ref_in,x,T_in);    
    return y, p, x

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

def forward_model_batched(points, transducers):
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
    
    bessel_arg=k*radius*torch.divide(planar_distance,distance)
    directivity=1/2-torch.pow(bessel_arg,2)/16+torch.pow(bessel_arg,4)/384
    
    p = 1j*k*distance
    phase = torch.e**(p)

    trans_matrix=2*P_ref*torch.multiply(torch.divide(phase,distance),directivity)

    return trans_matrix.permute((0,2,1))

def naive_solver_batch(points,board):
    '''
    Batched naive (backpropagation) algorithm for phase retrieval\\
    `points` Target point positions\\
    `board` The Transducer array, default two 16x16 arrays\\
    returns (point activations, hologram)
    '''
    activation = torch.ones(points.shape[2],1) +0j
    activation = activation.to(device)
    forward = forward_model_batched(points,board)
    back = torch.conj(forward).mT
    trans = back@activation
    trans_phase=  trans / torch.abs(trans)
    out = forward@trans_phase

    return out, trans_phase

def wgs_batch(A, b, iterations):
    '''
    batched WGS solver for transducer phases\\
    `A` Forward model matrix to use \\ 
    `b` initial guess - normally use `torch.ones(self.N,1).to(device)+0j`\\
    `iterations` number of iterations to run for \\
    returns (hologram image, point phases, hologram)
    '''
    AT = torch.conj(A).mT.to(device)
    b0 = b.to(device)
    x = torch.ones(A.shape[2],1).to(device) + 0j
    for kk in range(iterations):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return y, p, x

N=2
p = create_points(N,1)
A = forward_model_batched(p,TRANSDUCERS)
_,x_naive = naive_solver_batch(p,TRANSDUCERS)
print(torch.abs(A@x_naive))


_,_,x_wgs = wgs_batch(A, torch.ones(N,1).to(device)+0j,200)
print(torch.abs(A@x_wgs))


T_in = torch.pi/64 #Hologram phase change threshold
T_out = 0 #Point activations phase change threshold

p = p + 0.0005 #Move particles a small amount - 0.5mm
A = forward_model_batched(p,TRANSDUCERS)
_,_,x = temporal_wgs(A,torch.ones(N,1).to(device)+0j,200, x_wgs, A@x_wgs, T_in, T_out)
print(torch.abs(A@x))
