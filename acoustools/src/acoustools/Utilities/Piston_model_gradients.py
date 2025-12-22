from acoustools.Utilities.Boards import TRANSDUCERS
from acoustools.Utilities.Setup import device, DTYPE
import acoustools.Constants as Constants

import torch
from torch import Tensor


def compute_gradients(points, transducers = TRANSDUCERS, p_ref = Constants.P_ref, k=Constants.k, transducer_radius = Constants.radius):
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

    if type(p_ref) == float or type(p_ref) == int:
        p_ref = torch.ones(1,M,1, device=device, dtype=DTYPE) * p_ref

    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    diff =  points - transducers
    distances = torch.sqrt(torch.sum(diff**2, 2))
    planar_distance= torch.sqrt(torch.sum((diff**2)[:,:,0:2,:],dim=2))
    planar_distance = planar_distance +  1e-8

    #Partial derivates of bessel function section wrt xyz
    sin_theta = torch.divide(planar_distance,distances) 
    partialFpartialU = -1* (k**2 * transducer_radius**2)/4 * sin_theta + (k**4 * transducer_radius**4)/48 * sin_theta**3
    partialUpartiala = torch.ones_like(diff)
    
    diff_z = torch.unsqueeze(diff[:,:,2,:],2)
    diff_z = diff_z.expand((-1,-1,2,-1))
    
    denom = torch.unsqueeze((planar_distance*distances**3),2)
    denom = denom.expand((-1,-1,2,-1))
    # denom[denom == 0] = 1
    
    partialUpartiala[:,:,0:2,:] = -1 * (diff[:,:,0:2,:] * diff_z**2) / denom
    partialUpartiala[:,:,2,:] = (diff[:,:,2,:] * planar_distance) / distances**3

    partialFpartialU = torch.unsqueeze(partialFpartialU,2)
    partialFpartialU = partialFpartialU.expand((-1,-1,3,-1))
    partialFpartialX  = partialFpartialU * partialUpartiala

    #Grad of Pref / d(xt,t)
    dist_expand = torch.unsqueeze(distances,2)
    dist_expand = dist_expand.expand((-1,-1,3,-1))

    partialGpartialX = (p_ref.unsqueeze(3) * diff) / dist_expand**3

    #Grad of e^ikd(xt,t)
    partialHpartialX = 1j * k * (diff / dist_expand) * torch.exp(1j * k * dist_expand)

    #Combine
    bessel_arg=k*transducer_radius*torch.divide(planar_distance,distances)
    F=1-torch.pow(bessel_arg,2)/8+torch.pow(bessel_arg,4)/192
    F = torch.unsqueeze(F,2)
    F = F.expand((-1,-1,3,-1))

    G = p_ref.unsqueeze(3) / dist_expand
    H = torch.exp(1j * k * dist_expand)


    return F,G,H, partialFpartialX, partialGpartialX, partialHpartialX, partialFpartialU, partialUpartiala

def forward_model_grad(points:Tensor, transducers:Tensor|None = None, p_ref=Constants.P_ref, k=Constants.k, transducer_radius=Constants.radius) -> tuple[Tensor]:
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

    F,G,H, partialFpartialX, partialGpartialX, partialHpartialX,_,_ = compute_gradients(points, transducers, p_ref=p_ref,k=k, transducer_radius=transducer_radius)
    derivative = G*(H*partialFpartialX + F*partialHpartialX) + F*H*partialGpartialX
    derivative = derivative.to(device).to(DTYPE) # minus here to match f.d -> not 100% sure why its needed


    return derivative[:,:,0,:].permute((0,2,1)), derivative[:,:,1,:].permute((0,2,1)), derivative[:,:,2,:].permute((0,2,1))


def forward_model_second_derivative_unmixed(points:Tensor, transducers:Tensor|None = None, p_ref = Constants.P_ref, k=Constants.k, transducer_radius=Constants.radius) ->Tensor:
    '''
    Computes the second degree unmixed analytical gradient of the piston model\n
    :param points: Point position to compute propagation to 
    :param transducers: The Transducer array, default two 16x16 arrays 
    :return: second degree unmixed derivatives of forward model wrt x,y,z position Pxx, Pyy, Pzz
    '''

    #See Bk.2 Pg.314

    if transducers is None:
        transducers= TRANSDUCERS
    
    if type(p_ref) == float or type(p_ref) == int:
        M = transducers.shape[0]
        p_ref = torch.ones(1,M,1, device=device, dtype=DTYPE) * p_ref

    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]
    
    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    diff = transducers - points
    
    diff_square = diff**2
    distances = torch.sqrt(torch.sum(diff_square, 2))
    distances_square = distances ** 2
    distances_cube = distances ** 3
    distances_five = distances ** 5
    

    distances_expanded = distances.unsqueeze(2).expand((B,M,3,N))
    distances_expanded_square = distances_expanded**2
    distances_expanded_cube = distances_expanded ** 3
    
    planar_distance= torch.sqrt(torch.sum(diff_square[:,:,0:2,:],dim=2))
    planar_distance = planar_distance + 1e-8
    planar_distance_square = planar_distance**2

    sin_theta = planar_distance / distances
    sin_theta_expand = sin_theta.unsqueeze(2).expand((B,M,3,N))
    sin_theta_expand_square = sin_theta_expand**2

    p_ref_expand = p_ref.unsqueeze(2).expand((B,-1,3,N))

    dx = diff[:,:,0,:]
    dy = diff[:,:,1,:]
    dz = diff[:,:,2,:]

    # F = G * H 
    # G  = Pref * e^(ikd) / d
    # H = 1 - (kr sin(theta))^2 / 8 + (kr sin(theta))^4 / 192

    G = p_ref * torch.exp(1j * k * distances) / distances

    kr = k * transducer_radius
    kr_sine = kr*sin_theta
    H = 1 - ((kr_sine)**2) / 8 + ((kr_sine)**4)/192 


    #(a = {x,y,z})
    #Faa = 2*Ga*Ha + Gaa * H + G * Haa

    #Ga = Pref * [i*da * e^{ikd} * (kd+i) / d^2]

    #d = distance
    #da = -(at - a)^2 / d

    da = -1 * diff / distances_expanded
    kd = k * distances_expanded
    phase = torch.exp(1j*kd)
    Ga = p_ref_expand.expand((B,-1,3,N)) * ( (1j*da*phase * (kd + 1j))/ (distances_expanded_square))

    #Gaa = Pref * [ -1/d^3 * e^{ikd} * (da^2 * (k^2*d^2 + 2ik*d - 2) + d*daa * (1-ikd))]
    #daa = distance_bs / d^3
    # distance_bs = sum(b_t - b)^2 . b = {x,y,z} \ a
    distance_xy = diff[:,:,0,:] **2 + diff[:,:,1,:] **2
    distance_xz = diff[:,:,0,:] **2 + diff[:,:,2,:] **2
    distance_yz = diff[:,:,1,:] **2 + diff[:,:,2,:] **2

    distance_bs = torch.stack([distance_yz,distance_xz,distance_xy], dim =2)
    daa = distance_bs / distances_expanded_cube

    Gaa =  p_ref_expand * (-1/distances_expanded_cube * torch.exp(1j*kd) * (da**2 * (kd**2 + 2*1j*kd - 2) + distances_expanded *daa * (1-1j * kd)))

    #Ha = (kr)^2/48 * s * sa * ((kr)^2 * s^2 - 12)
    #s = planar_distance / distance = sin_theta
    #sb = -1 * (db * dz^2) / (sqrt(dx^2+dy^2) * distance^3). b = {x,y}
    #sz = (dz * sqrt(dx^2 + dy^2)) / distance^3

    sx = -1 * (dx * dz**2) / (planar_distance * distances_cube)
    sy = -1 * (dy * dz**2) / (planar_distance * distances_cube)
    sz = (dz * planar_distance) / distances_cube
    sa = torch.stack([sx,sy,sz],axis=2)
    # sa[sa.isnan()] = 1

    Ha = 1/48 * kr**2 * sin_theta_expand * sa * (kr**2 * sin_theta_expand**2 - 12)

    #Haa = 1/48 * (kr)^2 * (3*sa^2 * ((kr)^2 * s^2 - 4 ) + s * saa * ((kr)^2 * s^2 - 12))

    #sbb = [ dz^2 [ -1 * (db^2 * distance ^2) + (planar_distance^2 * distance^2) - 3*db^2 * planar_distance^2]] / planar_distance ^3 * distance ^5
    #szz = (-1 * planar_distance * (planar_distance^2 - 2*dz^2)) / distances^5  
    sxx = (dz**2 * (-1 * (dx**2 * distances_square) + (planar_distance_square * distances_square) - 3*dx**2 * planar_distance_square)) / (planar_distance**3 * distances_five)
    syy = (dz**2 * (-1 * (dy**2 * distances_square) + (planar_distance_square * distances_square) - 3*dy**2 * planar_distance_square)) / (planar_distance**3 * distances_five)
    szz = ((-1 * planar_distance) * (planar_distance**2 - 2*dz**2)) / distances_five
    saa = torch.stack([sxx,syy,szz],axis=2)
    # saa[saa.isnan()] = 0

    Haa = 1/48 * kr**2 * (3*sa**2 * (kr**2 * sin_theta_expand_square- 4) + sin_theta_expand*saa * (kr**2*sin_theta_expand_square - 12))


    H_expand = H.unsqueeze(2).expand((B,M,3,N))
    G_expand = G.unsqueeze(2).expand((B,M,3,N))
    Faa = 2*Ga*Ha + Gaa*H_expand + G_expand*Haa
    Faa = Faa.to(device).to(DTYPE)

    

    return Faa[:,:,0,:].permute((0,2,1)), Faa[:,:,1,:].permute((0,2,1)), Faa[:,:,2,:].permute((0,2,1))

def forward_model_second_derivative_mixed(points: Tensor, transducers:Tensor|None = None, p_ref = Constants.P_ref,  k=Constants.k, transducer_radius=Constants.radius)->Tensor:
    '''
    Computes the second degree mixed analytical gradient of the piston model\n
    :param points: Point position to compute propagation to 
    :param transducers: The Transducer array, default two 16x16 arrays 
    Returns second degree mixed derivatives of forward model wrt x,y,z position - Pxy, Pxz, Pyz
    '''

    #Bk.2 Pg.317

    if type(p_ref) == float or type(p_ref) == int:
        M = transducers.shape[0]
        p_ref = torch.ones(1,M,1, device=device, dtype=DTYPE) * p_ref

    if transducers is None:
        transducers= TRANSDUCERS

    B = points.shape[0]
    N = points.shape[2]
    M = transducers.shape[0]
    
    transducers = torch.unsqueeze(transducers,2)
    transducers = transducers.expand((B,-1,-1,N))
    points = torch.unsqueeze(points,1)
    points = points.expand((-1,M,-1,-1))

    diff = transducers - points
    
    diff_square = diff**2
    distances = torch.sqrt(torch.sum(diff_square, 2))
    distances_cube = distances ** 3
    distances_five = distances ** 5
    
    distances_expanded = distances.unsqueeze(2).expand((B,M,3,N))
    distances_expanded_square = distances_expanded**2
    
    planar_distance= torch.sqrt(torch.sum(diff_square[:,:,0:2,:],dim=2))
    planar_distance = planar_distance +  1e-8
    planar_distance_cube = planar_distance**3

    sin_theta = planar_distance / distances
    sin_theta_expand = sin_theta.unsqueeze(2).expand((B,M,3,N))

    dx = diff[:,:,0,:]
    dy = diff[:,:,1,:]
    dz = diff[:,:,2,:]

    p_ref_expand = p_ref.unsqueeze(2).expand((B,-1,3,N))

    # F = G * H 
    # G  = Pref * e^(ikd) / d
    # H = 1 - (kr sin(theta))^2 / 8 + (kr sin(theta))^4 / 192

    G = p_ref * torch.exp(1j * k * distances) / distances

    kr = k * transducer_radius
    kr_sine = kr*sin_theta
    H = 1 - ((kr_sine)**2) / 8 + ((kr_sine)**4)/192 

    #(a = {x,y,z})

    #Ga = Pref * [i*da * e^{ikd} * (kd+i) / d^2]

    #d = distance
    #da = -(at - a)^2 / d

    da = -1 * diff / distances_expanded
    dax = da[:,:,0,:]
    day = da[:,:,1,:]
    daz = da[:,:,2,:]


    kd_exp = k * distances_expanded
    kd = k * distances
    phase = torch.exp(1j*kd_exp)
    Ga = p_ref_expand * ( (1j*da*phase * (kd_exp + 1j))/ (distances_expanded_square))

    #Ha = (kr)^2/48 * s * sa * ((kr)^2 * s^2 - 12)
    #s = planar_distance / distance = sin_theta
    #sb = -1 * (db * dz^2) / (sqrt(dx^2+dy^2) * distance^3). b = {x,y}
    #sz = (dz * sqrt(dx^2 + dy^2)) / distance^3

    sx = -1 * (dx * dz**2) / (planar_distance * distances_cube)
    sy = -1 * (dy * dz**2) / (planar_distance * distances_cube)
    sz = (dz * planar_distance) / distances_cube

    # sx[sx.isnan()] = 1
    # sy[sy.isnan()] = 1
    # sz[sx.isnan()] = 1
    sa = torch.stack([sx,sy,sz],axis=2)
    

    Ha = 1/48 * kr**2 * sin_theta_expand * sa * (kr**2 * sin_theta_expand**2 - 12)

    #Gab = P_ref * e^{ikd} * (db * da * ( (kd)^2 + 2ikd - 2) + d * dab * (1-ikd)) / (-1*d^3)
    #dab = -da*db / d^3

    dxy = -1*dx*dy / distances_cube
    dxz = -1*dx*dz / distances_cube
    dyz = -1*dy*dz / distances_cube


    Gxy = (p_ref * torch.exp(1j * kd) * (day * dax * (kd**2 + 2*1j*kd - 2) + distances * dxy * (1 - 1j*kd))) / (-1 * distances_cube)
    Gxz = (p_ref * torch.exp(1j * kd) * (daz * dax * (kd**2 + 2*1j*kd - 2) + distances * dxz * (1 - 1j*kd))) / (-1 * distances_cube)
    Gyz = (p_ref * torch.exp(1j * kd) * (day * daz * (kd**2 + 2*1j*kd - 2) + distances * dyz * (1 - 1j*kd))) / (-1 * distances_cube)

    #Hab = (kr)^2/ 48 * (3*Sb*Sa * ((kr)^2 S^2 - 4) + S*Sab*((kr)^2 S^2 - 12))

    #Sxy = -dx * dy * dz^2 ( 4 * (dx^2 + dy^2) + dz^2 ) / (dx^2 + dy^2)^(3/2) * d^5
    #Saz = da * dz * (2 * dx**2 + 2 * dy**2 - dz**2) / (dx^2 + dy^2)^(1/2) * d^5

    Sxy = -1 * dx * dy * dz**2 * (4 * (dx**2 + dy**2) + dz**2) / (planar_distance_cube* distances_five)
    Sxz = dx * dz * (2 * dx**2 + 2 * dy**2 - dz**2) / (planar_distance * distances_five)
    Syz = dy * dz * (2 * dx**2 + 2 * dy**2 - dz**2) / (planar_distance * distances_five)
    # Sxy[Sxy.isnan()] = 1
    # Sxz[Sxz.isnan()] = 1
    # Syz[Syz.isnan()] = 1

    Hxy = kr**2 / 48 * (3 * sx * sy * (kr**2 * sin_theta**2 - 4) + sin_theta * Sxy * (kr**2 * sin_theta**2  -12))
    Hxz = kr**2 / 48 * (3 * sx * sz * (kr**2 * sin_theta**2 - 4) + sin_theta * Sxz * (kr**2 * sin_theta**2  -12))
    Hyz = kr**2 / 48 * (3 * sy * sz * (kr**2 * sin_theta**2 - 4) + sin_theta * Syz * (kr**2 * sin_theta**2  -12))


    #Fab = Ga*Hb + Gb*Ha + Gab * H + G * Hab

    Gx = Ga[:,:,0,:]
    Gy = Ga[:,:,1,:]
    Gz = Ga[:,:,2,:]

    Hx = Ha[:,:,0,:]
    Hy = Ha[:,:,1,:]
    Hz = Ha[:,:,2,:]


    Fxy = Gx * Hy + Gy * Hx + Gxy * H + G*Hxy
    Fxz = Gx * Hz + Gz * Hx + Gxz * H + G*Hxz
    Fyz = Gy * Hz + Gz * Hy + Gyz * H + G*Hyz

    Fxy = Fxy.to(device).to(DTYPE)
    Fxz = Fxz.to(device).to(DTYPE)
    Fyz = Fyz.to(device).to(DTYPE)

    return Fxy.permute((0,2,1)), Fxz.permute((0,2,1)), Fyz.permute((0,2,1))