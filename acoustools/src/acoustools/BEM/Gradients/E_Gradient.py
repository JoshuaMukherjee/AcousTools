
import torch
from torch import Tensor

from vedo import Mesh

import hashlib, pickle

from acoustools.Utilities import DTYPE, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed
from acoustools.BEM.Forward_models import get_cache_or_compute_H
from acoustools.Mesh import get_areas, get_centres_as_points, get_normals_as_points
import acoustools.Constants as Constants


 
# def get_G_partial(points:Tensor, scatterer:Mesh, board:None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:
#     '''
#     Computes gradient of the G matrix in BEM \n
#     :param points: Points to propagate to
#     :param scatterer: The mesh used (as a `vedo` `mesh` object)
#     :param board: Ignored
#     :param return_components: if true will return the subparts used to compute
#     :return: Gradient of the G matrix in BEM
#     '''
#     #Bk3. Pg. 26

#     areas = get_areas(scatterer)
#     centres = get_centres_as_points(scatterer)
#     normals = get_normals_as_points(scatterer)

#     N = points.shape[2]
#     M = centres.shape[2]

#     points = points.unsqueeze(3).expand(-1,-1,-1,M)
#     centres = centres.unsqueeze(2).expand(-1,-1,N,-1)

#     diff = centres - points    
#     diff_square = diff**2
#     diff_square_sum = torch.sum(diff_square, 1)
#     distances = torch.sqrt(torch.sum(diff_square, 1))
#     distances_expanded = distances.unsqueeze(1).expand((1,3,N,M))
#     distances_expanded_square = distances_expanded**2
#     distances_expanded_cube = distances_expanded**3

#     # G  =  e^(ikd) / 4pi d
#     G = areas * torch.exp(1j * Constants.k * distances_expanded) / (4*3.1415*distances_expanded)

#     #Ga =  [i*da * e^{ikd} * (kd+i) / 4pi d^2]

#     #d = distance
#     #da = -(at - a)^2 / d

#     da = -diff / distances_expanded
#     kd = Constants.k * distances_expanded
#     phase = torch.exp(1j*kd)
#     # Ga =  areas * ( (1j*diff*phase * (kd - 1j))/ (4*3.1415*distances_expanded_cube))
#     print(diff)
#     Ga = (areas * diff * phase * (1-1j*Constants.k*distances_expanded)) / (4*Constants.pi * distances_expanded_cube)

#     #P = (ik - 1/d)
#     P = (1j*Constants.k - 1/distances_expanded)
#     #Pa = da / d^2
#     Pa = -da / distances_expanded_cube

#     #C = (diff \cdot normals) / distances

#     nx = normals[:,0]
#     ny = normals[:,1]
#     nz = normals[:,2]

#     dx = diff[:,0,:]
#     dy = diff[:,1,:]
#     dz = diff[:,2,:]

#     C = (nx*dx + ny*dy + nz*dz) / distances


#     distances_cubed = distances**3
#     distance_square = distances**2

#     # Cx = (nx*(dy**2 + dz**2) - dx * (ny*dy + nz*dz)) / distances_cubed
#     # Cy = (ny*(dx**2 + dz**2) - dy * (nx*dx + nz*dz)) / distances_cubed
#     # Cz = (nz*(dx**2 + dy**2) - dz * (nx*dx + ny*dy)) / distances_cubed

#     Cx = (dx * (nx*dx + ny*dy + nz*dz) - nx*(diff_square_sum) ) / distances_cubed
#     Cy = (dy * (nx*dx + ny*dy + nz*dz) - ny*(diff_square_sum) ) / distances_cubed
#     Cz = (dz * (nx*dx + ny*dy + nz*dz) - nz*(diff_square_sum) ) / distances_cubed

#     Cx.unsqueeze_(1)
#     Cy.unsqueeze_(1)
#     Cz.unsqueeze_(1)

#     Ca = torch.cat([Cx, Cy, Cz],axis=1)

#     grad_G = Ga*P*C + G*P*Ca + G*Pa*C

#     grad_G =  grad_G.to(DTYPE)

#     if return_components:
#         return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:], G,P,C,Ga,Pa,Ca
    
#     return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:]


def BEM_forward_model_grad(points:Tensor, scatterer:Mesh, transducers:Tensor=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H:Tensor|None=None, return_components:bool=False,
                           path:str="Media") -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    Computes the gradient of the forward propagation for BEM\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param transducers: Transducers to use, if `None` uses `acoustools.Utilities.TRANSDUCERS`
    :param use_cache_H_grad: If true uses the cache system, otherwise computes `H` and does not save it
    :param print_lines: if true prints messages detaling progress
    :param H: Precomputed `H` - if `None` `H` will be computed
    :param return_components: if true will return the subparts used to compute
    :param path: path to folder containing `BEMCache/` 
    :return: Ex, Ey, Ez
    '''
    if transducers is None:
        transducers = TRANSDUCERS

    B = points.shape[0]
    if H is None:
        H = get_cache_or_compute_H(scatterer,transducers,use_cache_H, path, print_lines)
    
    Fx, Fy, Fz  = forward_model_grad(points, transducers)
    Gx, Gy, Gz = get_G_partial(points, scatterer, transducers)


    Ex = Fx - Gx@H #Minus coming out of the PM grad -> Not entoerly sure where that comes from but seems to fix it?
    Ey = Fy - Gy@H
    Ez = Fz - Gz@H


    if return_components:
        return Ex.to(DTYPE), Ey.to(DTYPE), Ez.to(DTYPE), Fx, Fy, Fz, Gx, Gy, Gz, H
    else:
        return Ex.to(DTYPE), Ey.to(DTYPE), Ez.to(DTYPE)
    

def get_G_second_unmixed(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:

    #Bk.3 Pg.80
    #P is much higher than the others?

    areas = get_areas(scatterer)
    centres = get_centres_as_points(scatterer)
    normals = get_normals_as_points(scatterer).unsqueeze(2)

    N = points.shape[2]
    M = centres.shape[2]

    points.requires_grad_()
    grad_Gx, grad_Gy, grad_Gz, G,P,C,Ga,Pa,Ca = get_G_partial(points, scatterer, board, True)
    # exit()

    points  = points.unsqueeze(3)  # [B, 3, N, 1]
    centres = centres.unsqueeze(2)  # [B, 3, 1, M]
    

    diff = points - centres    
    diff_square = diff**2
    diff_square_sum = torch.sum(diff_square, 1)
    distances = torch.sqrt(diff_square_sum)
    distances_square = distances ** 2
    distances_cube = distances ** 3
    distances_five = distances ** 5

    distances_expanded = distances.unsqueeze(1).expand((1,3,N,M))
    distances_expanded_square = distances_expanded**2
    distances_expanded_cube = distances_expanded**3
    distances_expanded_four = distances_expanded**4
    distances_expanded_five = distances_expanded**5

    distance_xy = diff[:,0,:,:] **2 + diff[:,1,:,:] **2
    distance_xz = diff[:,0,:,:] **2 + diff[:,2,:,:] **2
    distance_yz = diff[:,1,:,:] **2 + diff[:,2,:,:] **2


    da = diff / distances_expanded
    distance_bs = torch.stack([distance_yz,distance_xz,distance_xy], dim =1)
    daa = distance_bs / distances_expanded_cube

    kd = Constants.k * distances_expanded

    exp_ikd = torch.exp(1j*kd)
    
    # G  =  e^(ikd) / 4pi d
    # Gaa = (areas/4*Constants.pi) * (-1/distances_expanded_cube * torch.exp(1j*kd) * (da**2 * (kd**2 + 2*1j*kd - 2) + distances_expanded *daa * (1-1j * kd)))
    # Gaa = -areas / (4*Constants.pi * distances_expanded_cube) * exp_ikd * (da**2 * (kd**2 + 2j * kd - 2) + distances_expanded *daa * (1-1j * kd))
  
    # Gaa = areas * ((-Constants.k**2 * diff_square * exp_ikd) / diff_square_sum + (1j*Constants.k * exp_ikd) / distances - (1j * Constants.k * diff_square * exp_ikd)/ distances_expanded_cube) / (4 * Constants.pi * distances)
    # Gaa += areas * (((3*diff_square)/distances_expanded_five - 1/(distances_expanded_cube)) * exp_ikd) / (4 * Constants.pi)
    # Gaa -= (1j * Constants.k * areas * diff_square * exp_ikd) / (2 * Constants.pi * distances_expanded_four) #Same as Gaa above
    Gaa = areas/(4*Constants.pi * 1j) *exp_ikd * ((1j*Constants.k**2 * diff_square)/ distances_expanded_cube + (1j*Constants.k)/distances_expanded_square - 1/distances_expanded_cube - (3*1j*Constants.k*diff_square)/distances_expanded_four + (3*diff_square)/distances_expanded_five)
    Gxx = Gaa[:,0]
    Gyy = Gaa[:,1]
    Gzz = Gaa[:,2]
    
    #P = ik - 1/d
    # Paa = (distances_expanded * daa - 2*da**2) / distances_expanded_cube
    # Paa = (3 * diff_square - distances_expanded_square) / distances_expanded_five
    Paa = 1/distances_expanded_cube - (3*diff_square) / distances_expanded_five  #Equal to lines above
    Pxx = Paa[:,0]
    Pyy = Paa[:,1]
    Pzz = Paa[:,2]

    #C = (diff dot c )/ d

    dx = diff[:,0,:,:]
    dy = diff[:,1,:,:]
    dz = diff[:,2,:,:]


    nx = normals[:,0,:]
    ny = normals[:,1,:]
    nz = normals[:,2,:]

    nd = nx * dx + ny * dy + nz * dz

    Cxx = (2 * nx * dx)/distances_cube - nd/distances_cube + (3*nd*dx**2)/distances_five
    Cyy = (2 * ny * dy)/distances_cube - nd/distances_cube + (3*nd*dy**2)/distances_five
    Czz = (2 * nz * dz)/distances_cube - nd/distances_cube + (3*nd*dz**2)/distances_five

    # Caa = torch.stack([Cxx, Cyy, Czz], dim=1)

    # Cxx = (((3 * dax**2)/distances_five) - (1/distances_cube)) * nd - (2*nx*dx) / distances_cube
    # Cyy = (((3 * day**2)/distances_five) - (1/distances_cube)) * nd - (2*ny*dy) / distances_cube 
    # Czz = (((3 * daz**2)/distances_five) - (1/distances_cube)) * nd - (2*nz*dz) / distances_cube 
    # Caa = torch.stack([Cxx, Cyy, Czz], dim=1) 

    # grad_2_G_unmixed_old = 2 * Ca * (Ga * P + G * Pa) + C * (2 * Ga*Pa + Gaa * P + G * Paa) + G*P*Caa
    # grad_2_G_unmixed = C * (2*Ga * Pa + Gaa * P + G*Paa) + P * (2*Ga * Ca + G*Caa) + 2*G * Pa * Ca

    Cx = Ca[:,0]
    Cy = Ca[:,1]
    Cz = Ca[:,2]

    Px = Pa[:,0]
    Py = Pa[:,1]
    Pz = Pa[:,2]

    Gx = Ga[:,0]
    Gy = Ga[:,1]
    Gz = Ga[:,2]

    G = G[:,0,:]
    P = P[:,0,:] 
    


    grad_2_G_unmixed_xx = Gxx * P * C + G*Pxx*C + G*P*Cxx+  2*Gx*Px*C + 2*Gx*P*Cx + 2*G*Px*Cx
    grad_2_G_unmixed_yy = Gyy * P * C + G*Pyy*C + G*P*Cyy+  2*Gy*Py*C + 2*Gy*P*Cy + 2*G*Py*Cy
    grad_2_G_unmixed_zz = Gzz * P * C + G*Pzz*C + G*P*Czz+  2*Gz*Pz*C + 2*Gz*P*Cz + 2*G*Pz*Cz

    if return_components:
        return grad_2_G_unmixed_xx, grad_2_G_unmixed_yy, grad_2_G_unmixed_zz, G,P,C,Ga,Pa,Ca
    return grad_2_G_unmixed_xx, grad_2_G_unmixed_yy, grad_2_G_unmixed_zz


def get_G_second_mixed(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:

    #Bk.3 Pg.81
    
    areas = get_areas(scatterer).unsqueeze(1)
    centres = get_centres_as_points(scatterer)
    normals = get_normals_as_points(scatterer).unsqueeze(2)

    N = points.shape[2]
    M = centres.shape[2]

    grad_Gx, grad_Gy, grad_Gz, G,P,C,Ga,Pa,Ca = get_G_partial(points, scatterer, board, True)

    points  = points.unsqueeze(3)  # [B, 3, N, 1]
    centres = centres.unsqueeze(2)  # [B, 3, 1, M]

    diff = points - centres    
    diff_square = diff**2
    distances = torch.sqrt(torch.sum(diff_square, 1))
    distances_square = distances ** 2
    distances_cube = distances ** 3
    distances_four = distances ** 4
    distances_five = distances ** 5
    distances_six = distances ** 6
    
    kd = Constants.k * distances
    ikd = 1j * kd

    dx = diff[:,0,:,:]
    dy = diff[:,1,:,:]
    dz = diff[:,2,:,:]

    # print(dx.shape, dy.shape, dz.shape, distances_cube.shape)
    dxy = -dx*dy / distances_cube
    dxz = -dx*dz / distances_cube
    dyz = -dy*dz / distances_cube

    # print(dx.shape, distances.shape)
    dax = dx / distances
    day = dy / distances
    daz = dz / distances

    exp_ikd = torch.exp(ikd)



    # print(areas.shape, exp_ikd.shape, day.shape, dax.shape, distances.shape, distances_cube.shape)
    Gxy = (-areas * exp_ikd * (day * dax * (kd**2 + 2*ikd - 2) + distances * dxy * (1 - ikd))) / (4*Constants.pi * distances_cube)
    Gxz = (-areas * exp_ikd * (daz * dax * (kd**2 + 2*ikd - 2) + distances * dxz * (1 - ikd))) / (4*Constants.pi * distances_cube)
    Gyz = (-areas * exp_ikd * (day * daz * (kd**2 + 2*ikd - 2) + distances * dyz * (1 - ikd))) / (4*Constants.pi * distances_cube)

    # Gab = (areas/(Constants.pi*4j)) * exp_ikd * ((-1*Constants.k)/distances_cube - (3j * Constants.k)/distances_four + 3/distances_five )
   
    # print(exp_ikd.shape, distances_six.shape, distances.shape, dx.shape,dy.shape,dz.shape)
    # Gab = (areas/(Constants.pi*4j)) * exp_ikd * (1/distances_six * (3-1j*Constants.k*distances))
    # Gxy = (dx * dy) * Gab
    # Gxz = (dx * dz) * Gab
    # Gyz = (dz * dy) * Gab


    nx = normals[:,0,:]
    ny = normals[:,1,:]
    nz = normals[:,2,:]

    # print(nx.shape, dx.shape, ny.shape, dy.shape, nz.shape, dz.shape)
    nd = nx * dx + ny * dy + nz * dz

    # Cxy = (-ny*dx*distances_square - nx*dy*distances_square + 3 * dy*dx * nd) / distances_five
    # Cxz = (-nz*dx*distances_square - nx*dz*distances_square + 3 * dz*dx * nd) / distances_five
    # Cyz = (-ny*dz*distances_square - ny*dz*distances_square + 3 * dz*dy * nd) / distances_five

    # print(nx.shape, dy.shape, ny.shape, dx.shape, distances_cube.shape,distances_square.shape )
    Cxy = (-nx*dy - ny*dx + (3*nd*dx*dy/distances_cube)) / distances_square
    Cxz = (-nx*dz - nz*dx + (3*nd*dx*dz/distances_cube)) / distances_square
    Cyz = (-nz*dy - ny*dz + (3*nd*dz*dy/distances_cube)) / distances_square

    # Cxy = (2*ny*dx - nx*dy) / distances_cube - (3*(ny*distances_square - nd*dy)* dx) / distances_five
    # Cxz = (2*nz*dx - nx*dz) / distances_cube - (3*(nz*distances_square - nd*dz)* dx) / distances_five
    # Cyz = (2*ny*dz - nz*dy) / distances_cube - (3*(ny*distances_square - nd*dy)* dz) / distances_five
    
   
    # Pxy = -3 * (dx * dy) / distances_five
    # Pxz = -3 * (dx * dz) / distances_five
    # Pyz = -3 * (dz * dy) / distances_five

    Pxy = (distances * dxy - 2 * day * dax) / distances_cube
    Pxz = (distances * dxz - 2 * daz * dax) / distances_cube
    Pyz = (distances * dyz - 2 * day * daz) / distances_cube

    
    G = G[:,0,:]
    P = P[:,0,:] 
    # C = C.unsqueeze(1).expand(-1,3,-1,-1)

    Gx = Ga[:,0,:]
    Gy = Ga[:,1,:]
    Gz = Ga[:,2,:]

    Px = Pa[:,0,:]
    Py = Pa[:,1,:]
    Pz = Pa[:,2,:]

    Cx = Ca[:,0,:]
    Cy = Ca[:,1,:]
    Cz = Ca[:,2,:]


    # print(Gx*Py*C, G*Py*Cx)
    # print(Gxy*P*C , Gy*Px*C , Gy*P*Cx , Gx*Py*C , G*Pxy*C , G*Py*Cx , Gx*P*Cy , G*Px*Cy , G*P*Cxy, sep="\n\n")
    # print(G.shape, Gx.shape, Gy.shape, Gz.shape,P.shape , Px.shape, Py.shape, Pz.shape, C.shape, Cx.shape, Cy.shape, Cz.shape )

    # grad_2_G_mixed_xy = Gxy*P*C + Gy*Px*C + Gy*P*Cx + Gx*Py*C + G*Pxy*C + G*Py*Cx + Gx*P*Cy + G*Px*Cy + G*P*Cxy
    # grad_2_G_mixed_xz = Gxz*P*C + Gz*Px*C + Gz*P*Cx + Gx*Pz*C + G*Pxz*C + G*Pz*Cx + Gx*P*Cz + G*Px*Cz + G*P*Cxz
    # grad_2_G_mixed_yz = Gyz*P*C + Gy*Pz*C + Gy*P*Cz + Gz*Py*C + G*Pyz*C + G*Py*Cz + Gz*P*Cy + G*Pz*Cy + G*P*Cyz

    grad_2_G_mixed_xy = Gxy*P*C + Gx*Py*C + G*Pxy*C + G*Py*Cx + Gx*P*Cy + G*P*Cxy
    grad_2_G_mixed_xz = Gxz*P*C + Gx*Pz*C + G*Pxz*C + G*Pz*Cx + Gx*P*Cz + G*P*Cxz
    grad_2_G_mixed_yz = Gyz*P*C + Gz*Py*C + G*Pyz*C + G*Py*Cz + Gz*P*Cy + G*P*Cyz



    return grad_2_G_mixed_xy, grad_2_G_mixed_xz, grad_2_G_mixed_yz


def BEM_forward_model_second_derivative_unmixed(points:Tensor, scatterer:Mesh, transducers:Tensor=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H:Tensor|None=None, return_components:bool=False,
                           path:str="Media"):
    
    
    if transducers is None:
        transducers = TRANSDUCERS

    if H is None:
        H = get_cache_or_compute_H(scatterer,transducers,use_cache_H, path, print_lines)

    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=transducers)
    Gxx, Gyy, Gzz = get_G_second_unmixed(points, scatterer, transducers)

    Exx = Fxx + Gxx@H
    Eyy = Fyy + Gyy@H
    Ezz = Fzz + Gzz@H

    return Exx, Eyy, Ezz

def BEM_forward_model_second_derivative_mixed(points:Tensor, scatterer:Mesh, transducers:Tensor|Mesh=None, use_cache_H:bool=True, 
                           print_lines:bool=False, H:Tensor|None=None, return_components:bool=False,
                           path:str="Media"):
    
       
    if transducers is None:
        transducers = TRANSDUCERS

    if H is None:
        H = get_cache_or_compute_H(scatterer,transducers,use_cache_H, path, print_lines)

    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points,transducers=transducers)
    Gxy, Gxz, Gyz = get_G_second_mixed(points, scatterer, transducers)


    Exy = Fxy + Gxy@H
    Exz = Fxz + Gxz@H
    Eyz = Fyz + Gyz@H

    return Exy, Exz, Eyz





# def get_G_second_mixed(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:

#     #Bk.3 Pg.81
    
#     areas = get_areas(scatterer)
#     centres = get_centres_as_points(scatterer)
#     normals = get_normals_as_points(scatterer).unsqueeze(2)

#     N = points.shape[2]
#     M = centres.shape[2]

#     points = points.unsqueeze(3).expand(-1,-1,-1,M)
#     centres = centres.unsqueeze(2).expand(-1,-1,N,-1)

#     diff = points - centres    
#     diff_square = diff**2
#     distances = torch.sqrt(torch.sum(diff_square, 1))
#     distances_square = distances ** 2
#     distances_cube = distances ** 3
#     distances_four = distances ** 4
#     distances_five = distances ** 5

    
#     distances_expanded = distances.unsqueeze(1).expand((1,3,N,M)) 

#     kd = Constants.k * distances
#     exp_ikd = torch.exp(1j*kd)


#     dx = diff[:,0,:,:]
#     dy = diff[:,1,:,:]
#     dz = diff[:,2,:,:]

#     dxy = -dx*dy / distances_cube
#     dxz = -dx*dz / distances_cube
#     dyz = -dy*dz / distances_cube

#     da = diff / distances_expanded
#     dax = da[:,0,:,:]
#     day = da[:,1,:,:]
#     daz = da[:,2,:,:]
 
#     F = (areas * 1j * Constants.k * exp_ikd) / ( 4 * Constants.pi * distances)
#     P = (-1 * areas * exp_ikd) / ( 4 * Constants.pi * distances_square)

#     Fa = (-1 * Constants.k * areas * da * exp_ikd * (kd + 1j)) / (Constants.pi * 4 * distances_square)
#     Pa = (areas * da * exp_ikd * (2 - 1j*kd)) / (Constants.pi * 4 * distances_cube)


#     Fxy = (Constants.k * areas * exp_ikd)/(4 * Constants.pi * distances_cube) * (day * dax * (-1j * kd**2 + 2*kd + 2j) - distances*dxy * (kd + 1j))
#     Fxz = (Constants.k * areas * exp_ikd)/(4 * Constants.pi * distances_cube) * (daz * dax * (-1j * kd**2 + 2*kd + 2j) - distances*dxz * (kd + 1j))
#     Fyz = (Constants.k * areas * exp_ikd)/(4 * Constants.pi * distances_cube) * (day * daz * (-1j * kd**2 + 2*kd + 2j) - distances*dyz * (kd + 1j))

#     Pxy = (areas * exp_ikd)/(4 * Constants.pi * distances_four) * (day * dax * (kd**2 + 4j*kd -6) -distances*dxy * (2-1j*kd))
#     Pxz = (areas * exp_ikd)/(4 * Constants.pi * distances_four) * (daz * dax * (kd**2 + 4j*kd -6) -distances*dxz * (2-1j*kd))
#     Pyz = (areas * exp_ikd)/(4 * Constants.pi * distances_four) * (day * daz * (kd**2 + 4j*kd -6) -distances*dyz * (2-1j*kd))


#     nx = normals[:,0,:]
#     ny = normals[:,1,:]
#     nz = normals[:,2,:]

#     nd = nx * dx + ny * dy + nz * dz

#     C = (diff * normals).sum(dim=1) / distances

#     Cxy = (-ny*dx*distances_square - nx*dy*distances_square - 3 * dy*dx * nd) / distances_five
#     Cxz = (-nz*dx*distances_square - nx*dz*distances_square - 3 * dz*dx * nd) / distances_five
#     Cyz = (-ny*dz*distances_square - ny*dz*distances_square - 3 * dz*dy * nd) / distances_five


#     Cx = (nx*(dx**2 +dy**2 + dz**2) - dx * (ny*dy + nz*dz)) / distances_cube
#     Cy = (ny*(dx**2 +dy**2 + dz**2) - dy * (nx*dx + nz*dz)) / distances_cube
#     Cz = (nz*(dx**2 +dy**2 + dz**2) - dz * (nx*dx + ny*dy)) / distances_cube

#     Fx = Fa[:,0,:]
#     Fy = Fa[:,1,:]
#     Fz = Fa[:,2,:]

#     Px = Pa[:,0,:]
#     Py = Pa[:,1,:]
#     Pz = Pa[:,2,:]

#     print(Fx.shape, Fxy.shape)
#     print(Px.shape, Pxy.shape )
#     print(Cx.shape, Cxy.shape, C.shape)


#     grad_2_G_mixed_xy = Cy * (Fx + Px) + Cx * (Fy + Py) + C * (Fxy + Pxy) + Cxy * (F+P)
#     grad_2_G_mixed_xz = Cz * (Fx + Px) + Cx * (Fz + Pz) + C * (Fxz + Pxz) + Cxz * (F+P)
#     grad_2_G_mixed_yz = Cy * (Fz + Pz) + Cz * (Fy + Py) + C * (Fyz + Pyz) + Cyz * (F+P)



#     return grad_2_G_mixed_xy, grad_2_G_mixed_xz, grad_2_G_mixed_yz

def get_G_partial(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:
    '''
    Computes gradient of the G matrix in BEM \n
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Ignored
    :param return_components: if true will return the subparts used to compute
    :return: Gradient of the G matrix in BEM
    '''
    #Bk3. Pg. 26
    # if board is None:
    #     board = TRANSDUCERS

    areas = get_areas(scatterer)
    centres = get_centres_as_points(scatterer)
    normals = get_normals_as_points(scatterer)


    N = points.shape[2]
    M = centres.shape[2]


    # points = points.unsqueeze(3).expand(-1,-1,-1,M)
    # centres = centres.unsqueeze(2).expand(-1,-1,N,-1)
    points  = points.unsqueeze(3)  # [B, 3, N, 1]
    centres = centres.unsqueeze(2)  # [B, 3, 1, M]

    diff = points - centres    
    diff_square = diff**2
    distances = torch.sqrt(torch.sum(diff_square, 1))
    distances_expanded = distances.unsqueeze(1)#.expand((1,3,N,M))
    distances_expanded_square = distances_expanded**2
    distances_expanded_cube = distances_expanded**3

    # G  =  e^(ikd) / 4pi d
    G = areas * torch.exp(1j * Constants.k * distances_expanded) / (4*3.1415*distances_expanded)

    #Ga =  [i*da * e^{ikd} * (kd+i) / 4pi d^2]

    #d = distance
    #da = -(at - a)^2 / d
    da = diff / distances_expanded
    kd = Constants.k * distances_expanded
    phase = torch.exp(1j*kd)
    Ga =  areas * ( (1j*da*phase * (kd + 1j))/ (4*3.1415*distances_expanded_square))

    #P = (ik - 1/d)
    P = (1j*Constants.k - 1/distances_expanded)
    #Pa = da / d^2
    # Pa = da / distances_expanded_square
    Pa = diff / distances_expanded_cube

    #C = (diff \cdot normals) / distances

    nx = normals[:,0]
    ny = normals[:,1]
    nz = normals[:,2]

    dx = diff[:,0,:]
    dy = diff[:,1,:]
    dz = diff[:,2,:]

    n_dot_d = nx*dx + ny*dy + nz*dz

    dax = da[:,0,:]
    day = da[:,1,:]
    daz = da[:,2,:]

    C = (nx*dx + ny*dy + nz*dz) / distances


    distances_cubed = distances**3
    distance_square = distances**2


    Cx = 1/distance_square * (nx * distances - (n_dot_d * dx) / distances)
    Cy = 1/distance_square * (ny * distances - (n_dot_d * dy) / distances)
    Cz = 1/distance_square * (nz * distances - (n_dot_d * dz) / distances)

    Cx.unsqueeze_(1)
    Cy.unsqueeze_(1)
    Cz.unsqueeze_(1)

    Ca = torch.cat([Cx, Cy, Cz],axis=1)

    grad_G = Ga*P*C + G*P*Ca + G*Pa*C

    grad_G =  grad_G.to(DTYPE)

    if return_components:
        return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:], G,P,C,Ga,Pa, Ca
    
    return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:]
