import vedo
import numpy as np
import torch
import pickle

import matplotlib.pyplot as plt

from acoustools.Utilities import device, TOP_BOARD, TRANSDUCERS, forward_model_batched, create_points, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed
import acoustools.Constants as Constants
from acoustools.Mesh import scatterer_file_name, load_scatterer, load_multiple_scatterers, get_centres_as_points, board_name, get_areas, get_normals_as_points

import hashlib

def compute_green_derivative(y,x,norms,B,N,M, return_components=False):
    distance = torch.sqrt(torch.sum((x - y)**2,dim=3))

    vecs = y-x
    norms = norms.expand(B,N,-1,-1)

    
    # norm_norms = torch.norm(norms,2,dim=3) # === 1
    vec_norms = torch.norm(vecs,2,dim=3)
    angles = torch.sum(norms*vecs,3) / (vec_norms)

    A = (torch.e**(1j*Constants.k*distance))/(4*torch.pi*distance)
    B = (1j*Constants.k - 1/(distance))
    partial_greens = A*B*angles
    partial_greens[partial_greens.isnan()] = 1

    if return_components:
        return partial_greens, A,B,angles
    
    return partial_greens

def compute_G(points, scatterer):
    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    B = points.shape[0]
    N = points.shape[2]
    M = areas.shape[0]
    areas = areas.expand((B,N,-1))

    #Compute the partial derivative of Green's Function

    #Firstly compute the distances from mesh points -> control points
    centres = torch.tensor(scatterer.cell_centers).to(device) #Uses centre points as position of mesh
    centres = centres.expand((B,N,-1,-1))
    
    # print(points.shape)
    # p = torch.reshape(points,(B,N,3))
    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    #Compute cosine of angle between mesh normal and point
    scatterer.compute_normals()
    norms = torch.tensor(scatterer.cell_normals).to(device)
  
    partial_greens = compute_green_derivative(centres,p,norms, B,N,M)
    
    G = areas * partial_greens
    return G

def compute_A(scatterer):

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)

    centres = torch.tensor(scatterer.cell_centers).to(device)
    m = centres
    M = m.shape[0]
    m = m.expand((M,M,3))

    m_prime = m.clone()
    m_prime = m_prime.permute((1,0,2))

    norms = torch.tensor(scatterer.cell_normals).to(device)

    green = compute_green_derivative(m.unsqueeze_(0),m_prime.unsqueeze_(0),norms,1,M,M)
    # areas = areas.unsqueeze(0).T.expand((-1,M)).unsqueeze(0)
    A = green * areas * -1
    eye = torch.eye(M).to(bool)
    A[:,eye] = 0.5

    return A.to(torch.complex64)

def compute_bs(scatterer, board):
    centres = torch.tensor(scatterer.cell_centers).to(device).T.unsqueeze_(0)
    bs = forward_model_batched(centres,board)
    return bs.to(torch.complex64)

def compute_H(scatterer, board,use_LU=True):
    A = compute_A(scatterer)
    bs = compute_bs(scatterer,board)
    if not use_LU:
        H = torch.linalg.solve(A,bs)
    else:
        LU, pivots = torch.linalg.lu_factor(A)
        H = torch.linalg.lu_solve(LU, pivots, bs)

    return H

def grad_H(points, scatterer, transducers, return_components = False):
    '''
    Ignores `points` - for compatability with other gradient functions, takes centres of the scatterers
    '''
    centres = torch.tensor(scatterer.cell_centers).to(device).T.unsqueeze_(0)

    M = centres.shape[2]

    B = compute_bs(scatterer,transducers)
    A = compute_A(scatterer)
    A_inv = torch.inverse(A).to(torch.complex128)
    
    Bx, By, Bz = forward_model_grad(centres, transducers)
    Bx = Bx.to(torch.complex128)
    By = By.to(torch.complex128)
    Bz = Bz.to(torch.complex128)


    Ax, Ay, Az =  get_G_partial(centres,scatterer,transducers)
    # Ax *= -1
    # Ay *= -1
    # Az *= -1
    
    Ax = (-1* Ax)
    Ay = (-1* Ay)
    Az = (-1* Az)
    # .to(torch.complex64)

    
    eye = torch.eye(M).to(bool)
    Ax[:,eye] = 0
    Ay[:,eye] = 0
    Az[:,eye] = 0

    
    A_inv_x = (-1*A_inv @ Ax @ A_inv).to(torch.complex64)
    A_inv_y = (-1*A_inv @ Ay @ A_inv).to(torch.complex64)
    A_inv_z = (-1*A_inv @ Az @ A_inv).to(torch.complex64)

    Hx = (A_inv_x@B) + (A_inv@Bx)
    Hy = (A_inv_y@B) + (A_inv@By)
    Hz = (A_inv_z@B) + (A_inv@By)

    Hx = Hx.to(torch.complex64)
    Hy = Hy.to(torch.complex64)
    Hz = Hz.to(torch.complex64)

    if return_components:
        return Hx, Hy, Hz, A, A_inv, Ax, Ay, Az
    else:
        return Hx, Hy, Hz

def grad_2_H(points, scatterer, transducers, A = None, A_inv = None, Ax = None, Ay = None, Az = None):
    '''
    Ignores `points` - for compatability with other gradient functions, takes centres of the scatterers
    '''

    centres = get_centres_as_points(scatterer)
    M = centres.shape[2]

    B = compute_bs(scatterer,transducers)

    Fx, Fy, Fz = forward_model_grad(centres, transducers)
    Fx = Fx.to(torch.complex64)
    Fy = Fy.to(torch.complex64)
    Fz = Fz.to(torch.complex64)
    Fa = torch.stack([Fx,Fy,Fz],dim=3)

    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(centres, transducers)
    Faa = torch.stack([Fxx,Fyy,Fzz],dim=3)

    F = forward_model_batched(centres, transducers)
    
    if A is None:
        A = compute_A(scatterer)
    
    if A_inv is None:
        A_inv = torch.inverse(A)
    
    if Ax is None or Ay is None or Az is None:
        Ax, Ay, Az = get_G_partial(centres,scatterer,transducers)
        eye = torch.eye(M).to(bool)
        Ax[:,eye] = 0
        Ay[:,eye] = 0
        Az[:,eye] = 0
        Ax = Ax.to(torch.complex64)
        Ay = Ay.to(torch.complex64)
        Az = Az.to(torch.complex64)
    Aa = torch.stack([Ax,Ay,Az],dim=3)

    
    A_inv_x = (-1*A_inv @ Ax @ A_inv).to(torch.complex64)
    A_inv_y = (-1*A_inv @ Ay @ A_inv).to(torch.complex64)
    A_inv_z = (-1*A_inv @ Az @ A_inv).to(torch.complex64)


    A_inv_a = torch.stack([A_inv_x,A_inv_y,A_inv_z],dim=3)

    m = centres.permute(0,2,1)
    m = m.expand((M,M,3))

    m_prime = m.clone()
    m_prime = m_prime.permute((1,0,2))

    vecs = m - m_prime
    vecs = vecs.unsqueeze(0)
    

    norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = norms.expand(1,M,-1,-1)

    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    vec_norms_cube = vec_norms**3
    vec_norms_five = vec_norms**5

    distance = torch.sqrt(torch.sum(vecs**2,dim=3))
    vecs_square = vecs **2
    distance_exp = torch.unsqueeze(distance,3)
    distance_exp = distance_exp.expand(-1,-1,-1,3)
    
    distance_exp_cube = distance_exp**3

    distaa = torch.zeros_like(distance_exp)
    distaa[:,:,:,0] = (vecs_square[:,:,:,1] + vecs_square[:,:,:,2]) 
    distaa[:,:,:,1] = (vecs_square[:,:,:,0] + vecs_square[:,:,:,2]) 
    distaa[:,:,:,2] = (vecs_square[:,:,:,1] + vecs_square[:,:,:,0])
    distaa = distaa / distance_exp_cube

    dista = vecs / distance_exp

    Aaa = (-1 * torch.exp(1j*Constants.k * distance_exp) * (distance_exp*(1-1j*Constants.k*distance_exp))*distaa + dista*(Constants.k**2 * distance_exp**2 + 2*1j*Constants.k * distance_exp -2)) / (4*torch.pi * distance_exp_cube)
    
    Baa = (distance_exp * distaa - 2*dista**2) / distance_exp_cube

    Caa = torch.zeros_like(distance_exp).to(device)

    vec_dot_norm = vecs[:,:,:,0]*norms[:,:,:,0]+vecs[:,:,:,1]*norms[:,:,:,1]+vecs[:,:,:,2]*norms[:,:,:,2]

    Caa[:,:,:,0] = ((( (3 * vecs[:,:,:,0]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,0]*norms[:,:,:,0]) / (norm_norms*vec_norms_cube**3))
    Caa[:,:,:,1] = ((( (3 * vecs[:,:,:,1]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,1]*norms[:,:,:,1]) / (norm_norms*vec_norms_cube**3))
    Caa[:,:,:,2] = ((( (3 * vecs[:,:,:,2]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,2]*norms[:,:,:,2]) / (norm_norms*vec_norms_cube**3))
    
    Gx, Gy, Gz, A_green, B_green, C_green, Aa_green, Ba_green, Ca_green = get_G_partial(centres, scatterer, transducers, return_components=True)

    Gaa = 2*Ca_green*(B_green*Aa_green + A_green*Ba_green) + C_green*(B_green*Aaa + 2*Aa_green*Ba_green + A_green*Baa)+ A_green*B_green*Caa
    Gaa = Gaa.to(torch.complex64)

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    areas = torch.unsqueeze(areas,0)
    areas = torch.unsqueeze(areas,0)
    areas = torch.unsqueeze(areas,3)

    Gaa = Gaa * areas
    # Gaa = torch.nan_to_num(Gaa)
    eye = torch.eye(Gaa.shape[2]).to(bool)
    Gaa[:,eye] = 0
    
    
    A_inv_a = A_inv_a.permute(0,3,2,1)
    Fa = Fa.permute(0,3,1,2)

    A_inv = A_inv.unsqueeze(1).expand(-1,3,-1,-1)
    Faa = Faa.permute(0,3,1,2)

    Fa = Fa.to(torch.complex64)
    Faa = Faa.to(torch.complex64)

    Gaa = Gaa.permute(0,3,2,1)
    Aa = Aa.permute(0,3,2,1)
    Aa = Aa.to(torch.complex64)

    X1 = A_inv_a @ Fa + A_inv @ Faa
    X2 = (A_inv @ (Aa @ A_inv @ Aa - Gaa)@A_inv) @ F
    X3 = A_inv_a@Fa


    Haa = X1 + X2 + X3
    
    return Haa

def get_cache_or_compute_H_2_gradients(scatterer,board,use_cache_H_grad=True, path="Media", print_lines=False):
    if use_cache_H_grad:
        
        f_name = scatterer.filename+"--"+ board_name(board)
        f_name = hashlib.md5(f_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  f_name +"_2grad"+ ".bin"

        try:
            if print_lines: print("Trying to load H 2 grads at", f_name ,"...")
            Haa = pickle.load(open(f_name,"rb"))
            Haa = Haa.to(device)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H grad 2...")
            Haa = grad_2_H(None, transducers=board, **{"scatterer":scatterer })
            f = open(f_name,"wb")
            pickle.dump(Haa,f)
            f.close()
    else:
        if print_lines: print("Computing H grad 2...")
        Haa = grad_2_H(None, transducers=board, **{"scatterer":scatterer })

    return Haa

def get_cache_or_compute_H_gradients(scatterer,board,use_cache_H_grad=True, path="Media", print_lines=False):
    if use_cache_H_grad:
        
        f_name = scatterer.filename +"--"+ board_name(board)
        f_name = hashlib.md5(f_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  f_name +"_grad"+ ".bin"

        try:
            if print_lines: print("Trying to load H grads at", f_name ,"...")
            Hx, Hy, Hz = pickle.load(open(f_name,"rb"))
            Hx = Hx.to(device)
            Hy = Hy.to(device)
            Hz = Hz.to(device)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H Grads...")
            Hx, Hy, Hz = grad_H(None, transducers=board, **{"scatterer":scatterer })
            f = open(f_name,"wb")
            pickle.dump((Hx, Hy, Hz),f)
            f.close()
    else:
        if print_lines: print("Computing H Grad...")
        Hx, Hy, Hz = grad_H(None, transducers=board, **{"scatterer":scatterer })

    return Hx, Hy, Hz

def get_cache_or_compute_H(scatterer,board,use_cache_H=True, path="Media", print_lines=False):

    if use_cache_H:
        
        f_name = scatterer.filename+"--"+ board_name(board)
        f_name = hashlib.md5(f_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  f_name + ".bin"

        try:
            if print_lines: print("Trying to load H at", f_name ,"...")
            H = pickle.load(open(f_name,"rb")).to(device)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H...")
            H = compute_H(scatterer,board)
            f = open(f_name,"wb")
            pickle.dump(H,f)
            f.close()
    else:
        if print_lines: print("Computing H...")
        H = compute_H(scatterer,board)

    return H

def compute_E(scatterer, points, board=TOP_BOARD, use_cache_H=True, print_lines=False, H=None,path="Media", return_components=False):
    if print_lines: print("H...")
    
    if H is None:
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines)
        
    if print_lines: print("G...")
    G = compute_G(points, scatterer).to(torch.complex64)
    
    if print_lines: print("F...")
    F = forward_model_batched(points,board)
    
    if print_lines: print("E...")

    E = F+G@H

    if return_components:
        return E.to(torch.complex64), F.to(torch.complex64), G.to(torch.complex64), H.to(torch.complex64)
    return E.to(torch.complex64)

def propagate_BEM(activations,points,scatterer=None,board=TOP_BOARD,H=None,E=None,path="Media"):
    if E is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E = compute_E(scatterer,points,board,H=H, path=path)
    
    out = E@activations
    return out

def propagate_BEM_pressure(activations,points,scatterer=None,board=TOP_BOARD,H=None,E=None, path="Media"):
    point_activations = propagate_BEM(activations,points,scatterer,board,H,E,path)
    pressures =  torch.abs(point_activations)
    return pressures


def get_G_partial(points, scatterer, board=TRANSDUCERS, return_components=False):
    '''
    Returns gradient of the G matrix in BEM
    '''
    #Bk1. Page 272
    areas = get_areas(scatterer)
    centres = get_centres_as_points(scatterer)

    N = points.shape[2]
    M = centres.shape[2]

    points = points.unsqueeze(3).expand(-1,-1,-1,M)
    centres = centres.unsqueeze(2).expand(-1,-1,N,-1)



    vecs = points - centres #Centres -> Points
    vecs = vecs.to(torch.double)
    distances = torch.sum(vecs**2)
    norms = get_normals_as_points(scatterer).real.to(torch.double).unsqueeze(2).expand(-1,-1,N,-1)

    vec_norm = torch.norm(vecs,2)
    angle = torch.einsum('ijkh,ijkh->ikh', vecs, norms).unsqueeze(1) / vec_norm
    angle_grad = -1*norms / vec_norm
    phase = torch.exp(1j * Constants.k * distances)

    grad_G = areas * (-1 * phase / (4*torch.pi*distances**3) * (vecs / distances * angle * (Constants.k**2 * distances**2 + 2j*Constants.k*distances - 2) + distances * angle_grad * (1-1j*Constants.k*distances)))
    grad_G = grad_G.to(torch.complex128)
    
    return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:]


def BEM_forward_model_grad(points, scatterer, transducers=TRANSDUCERS, use_cache_H=True, print_lines=False, H=None, return_components=False,path="Media"):
    B = points.shape[0]
    if H is None:
        H = get_cache_or_compute_H(scatterer,transducers,use_cache_H, path, print_lines)
    
    Fx, Fy, Fz  = forward_model_grad(points, transducers)
    Gx, Gy, Gz = get_G_partial(points, scatterer, transducers)


    Gx[Gx.isnan()] = 0
    Gy[Gy.isnan()] = 0
    Gz[Gz.isnan()] = 0

    Fx = Fx.to(torch.complex128)
    Fy = Fy.to(torch.complex128)
    Fz = Fz.to(torch.complex128)

    H = H.expand(B, -1, -1).to(torch.complex128)


    Ex = Fx + Gx@H
    Ey = Fy + Gy@H
    Ez = Fz + Gz@H


    if return_components:
        return Ex.to(torch.complex64), Ey.to(torch.complex64), Ez.to(torch.complex64), Fx, Fy, Fz, Gx, Gy, Gz, H
    else:
        return Ex.to(torch.complex64), Ey.to(torch.complex64), Ez.to(torch.complex64)
    
def BEM_forward_model_second_derivative_unmixed(points, scatterer, board=TRANSDUCERS, use_cache_H=True, print_lines=False, H=None, return_components=False,path="Media"):
    B = points.shape[0]
    N = points.shape[2]

    if H is None:
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines)
    
    centres = torch.tensor(scatterer.cell_centers).to(device)
    M = centres.shape[0]
    
    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    vecs = p-centres #Centres -> Points
    norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = norms.expand(B,N,-1,-1)

    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    vec_norms_cube = vec_norms**3
    vec_norms_five = vec_norms**5

    distance = torch.sqrt(torch.sum(vecs**2,dim=3))
    vecs_square = vecs **2
    distance_exp = torch.unsqueeze(distance,3)
    distance_exp = distance_exp.expand(-1,-1,-1,3)
    
    distance_exp_cube = distance_exp**3

    distaa = torch.zeros_like(distance_exp)
    distaa[:,:,:,0] = (vecs_square[:,:,:,1] + vecs_square[:,:,:,2]) 
    distaa[:,:,:,1] = (vecs_square[:,:,:,0] + vecs_square[:,:,:,2]) 
    distaa[:,:,:,2] = (vecs_square[:,:,:,1] + vecs_square[:,:,:,0])
    distaa = distaa / distance_exp_cube

    dista = vecs / distance_exp


    Aaa = (-1 * torch.exp(1j*Constants.k * distance_exp) * (distance_exp*(1-1j*Constants.k*distance_exp))*distaa + dista*(Constants.k**2 * distance_exp**2 + 2*1j*Constants.k * distance_exp -2)) / (4*torch.pi * distance_exp_cube)
    
    Baa = (distance_exp * distaa - 2*dista**2) / distance_exp_cube

    Caa = torch.zeros_like(distance_exp).to(device)

    vec_dot_norm = vecs[:,:,:,0]*norms[:,:,:,0]+vecs[:,:,:,1]*norms[:,:,:,1]+vecs[:,:,:,2]*norms[:,:,:,2]

    Caa[:,:,:,0] = ((( (3 * vecs[:,:,:,0]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,0]*norms[:,:,:,0]) / (norm_norms*vec_norms_cube**3))
    Caa[:,:,:,1] = ((( (3 * vecs[:,:,:,1]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,1]*norms[:,:,:,1]) / (norm_norms*vec_norms_cube**3))
    Caa[:,:,:,2] = ((( (3 * vecs[:,:,:,2]**2) / (vec_norms_five) - (1)/(vec_norms_cube))*(vec_dot_norm)) / norm_norms) - ((2*vecs[:,:,:,2]*norms[:,:,:,2]) / (norm_norms*vec_norms_cube**3))
    
    Gx, Gy, Gz, A, B, C, Aa, Ba, Ca = get_G_partial(points, scatterer, board, return_components=True)

    Gaa = 2*Ca*(B*Aa + A*Ba) + C*(B*Aaa + 2*Aa*Ba + A*Baa)+ A*B*Caa
    Gaa = Gaa.to(torch.complex64)

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    areas = torch.unsqueeze(areas,0)
    areas = torch.unsqueeze(areas,0)
    areas = torch.unsqueeze(areas,3)

    Gaa = Gaa * areas

    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points)

    Exx = Fxx + Gaa[:,:,:,0]@H
    Eyy = Fyy + Gaa[:,:,:,1]@H
    Ezz = Fzz + Gaa[:,:,:,2]@H

    if return_components:
        return Exx, Eyy, Ezz, Fxx, Fyy, Fzz, Gx, Gy, Gz, A, B, C, Aa, Ba, Ca, H
    else:    
        return Exx, Eyy, Ezz

def BEM_forward_model_second_derivative_mixed(points, scatterer, board=TRANSDUCERS, use_cache_H=True, print_lines=False, H=None,path="Media"):
    if H is None:
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines)
    
    Batch = points.shape[0]
    N = points.shape[2]
    centres = torch.tensor(scatterer.cell_centers).to(device)
    M = centres.shape[0]
    
    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    vecs = p-centres #Centres -> Points
    norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = norms.expand(Batch,N,-1,-1)

    distance = torch.sqrt(torch.sum(vecs**2,dim=3))
    distance_square = distance**2
    distance_cube = distance**3
    
    distance_exp = torch.unsqueeze(distance,3)
    distance_exp = distance_exp.expand(-1,-1,-1,3)
    distance_exp_square = distance_exp**2
    distance_exp_cube = distance_exp**3

    distances_ab = torch.zeros(Batch,N,M,3).to(device) #0 -> xy, 1 -> xz, 2 -> yz
    distances_ab[:,:,:,0] = vecs[:,:,:,0]*vecs[:,:,:,1] 
    distances_ab[:,:,:,1] = vecs[:,:,:,0]*vecs[:,:,:,2]
    distances_ab[:,:,:,2] = vecs[:,:,:,1]*vecs[:,:,:,2]
    distances_ab = distances_ab/distance_exp_cube

    distance_a = torch.zeros(Batch,N,M,3).to(device)
    distance_a[:,:,:,0] = vecs[:,:,:,0]
    distance_a[:,:,:,1] = vecs[:,:,:,0]
    distance_a[:,:,:,2] = vecs[:,:,:,1]
    distance_a  = distance_a / distance_exp_cube

    Aab_term_1 = (1/(4*torch.pi*distance_cube)) * torch.e**(1j *Constants.k*distance)
    Aab = torch.zeros(Batch,N,M,3).to(device) +0j #0 -> xy, 1 -> xz, 2 -> yz
    Aab[:,:,:,0] = -1 * Aab_term_1 * (distance_a[:,:,:,0] * distance_a[:,:,:,1] * (Constants.k**2 * 2 * distance_square + 1j * Constants.k * distance - 2) + distance * distances_ab[:,:,:,0] * (1-1j*Constants.k*distance))
    Aab[:,:,:,1] = -1 * Aab_term_1 * (distance_a[:,:,:,0] * distance_a[:,:,:,2] * (Constants.k**2 * 2 * distance_square + 1j * Constants.k * distance - 2) + distance * distances_ab[:,:,:,1] * (1-1j*Constants.k*distance))
    Aab[:,:,:,2] = -1 * Aab_term_1 * (distance_a[:,:,:,1] * distance_a[:,:,:,2] * (Constants.k**2 * 2 * distance_square + 1j * Constants.k * distance - 2) + distance * distances_ab[:,:,:,2] * (1-1j*Constants.k*distance))

    Bab = torch.zeros(Batch,N,M,3).to(device) +0j #0 -> xy, 1 -> xz, 2 -> yz
    Bab[:,:,:,0] = (distance*distances_ab[:,:,:,0] - 2*distance_a[:,:,:,0]*distance_a[:,:,:,1]) / (distance_cube)
    Bab[:,:,:,1] = (distance*distances_ab[:,:,:,1] - 2*distance_a[:,:,:,0]*distance_a[:,:,:,2]) / (distance_cube)
    Bab[:,:,:,2] = (distance*distances_ab[:,:,:,2] - 2*distance_a[:,:,:,1]*distance_a[:,:,:,2]) / (distance_cube)

    vec_norm_prod = vecs*norms

    norm_norms = torch.norm(norms,2,dim=3)
    vec_norms = torch.norm(vecs,2,dim=3)
    vec_norms_cube = vec_norms**3
    vec_norms_five = vec_norms**5
    
    denom_1 = norm_norms*vec_norms_cube
    denom_2 = norm_norms*vec_norms_five

    Cab = torch.zeros(Batch,N,M,3).to(device) +0j #0 -> xy, 1 -> xz, 2 -> yz
    Cab[:,:,:,0] = (2*vec_norm_prod[:,:,:,1] - vec_norm_prod[:,:,:,0])/denom_1 - ((3*vecs[:,:,:,1] * (norms[:,:,:,1]*(vecs[:,:,:,2]**2 + vecs[:,:,:,1]**2) - vecs[:,:,:,0]*(vec_norm_prod[:,:,:,2]+vec_norm_prod[:,:,:,1])))) / denom_2
    Cab[:,:,:,1] = (2*vec_norm_prod[:,:,:,2] - vec_norm_prod[:,:,:,0])/denom_1 - ((3*vecs[:,:,:,2] * (norms[:,:,:,2]*(vecs[:,:,:,1]**2 + vecs[:,:,:,2]**2) - vecs[:,:,:,0]*(vec_norm_prod[:,:,:,1]+vec_norm_prod[:,:,:,2])))) / denom_2
    Cab[:,:,:,2] = (2*vec_norm_prod[:,:,:,2] - vec_norm_prod[:,:,:,1])/denom_1 - ((3*vecs[:,:,:,2] * (norms[:,:,:,1]*(vecs[:,:,:,0]**2 + vecs[:,:,:,2]**2) - vecs[:,:,:,1]*(vec_norm_prod[:,:,:,0]+vec_norm_prod[:,:,:,2])))) / denom_2


    # Exx, Eyy, Ezz, Fxx, Fyy, Fzz, Gx, Gy, Gz, A, B, C, Aa, Ba, Ca, H = BEM_forward_model_second_derivative_unmixed(points, scatterer, board, use_cache_H, print_lines, H, return_components=True)
    Gx, Gy, Gz, A, B, C, Aa, Ba, Ca = get_G_partial(points, scatterer,board,True)

    Gxy = C[:,:,:,0]*(Aa[:,:,:,0]*Ba[:,:,:,1] + Aa[:,:,:,1]*Ba[:,:,:,0] + Aab[:,:,:,0]*B[:,:,:,0] + A[:,:,:,0]*Bab[:,:,:,0]) + B[:,:,:,0] * (Aa[:,:,:,0] * Ca[:,:,:,1] + Aa[:,:,:,1] * Ca[:,:,:,0] + A[:,:,:,0]*Cab[:,:,:,0]) + A[:,:,:,0] * (Ba[:,:,:,0]*Ca[:,:,:,1] + Ba[:,:,:,1]*Ca[:,:,:,0])
    Gxz = C[:,:,:,0]*(Aa[:,:,:,0]*Ba[:,:,:,2] + Aa[:,:,:,2]*Ba[:,:,:,0] + Aab[:,:,:,1]*B[:,:,:,0] + A[:,:,:,0]*Bab[:,:,:,1]) + B[:,:,:,0] * (Aa[:,:,:,0] * Ca[:,:,:,2] + Aa[:,:,:,2] * Ca[:,:,:,0] + A[:,:,:,0]*Cab[:,:,:,1]) + A[:,:,:,0] * (Ba[:,:,:,0]*Ca[:,:,:,2] + Ba[:,:,:,2]*Ca[:,:,:,0])
    Gyz = C[:,:,:,0]*(Aa[:,:,:,1]*Ba[:,:,:,2] + Aa[:,:,:,2]*Ba[:,:,:,1] + Aab[:,:,:,2]*B[:,:,:,0] + A[:,:,:,0]*Bab[:,:,:,2]) + B[:,:,:,0] * (Aa[:,:,:,1] * Ca[:,:,:,2] + Aa[:,:,:,2] * Ca[:,:,:,1] + A[:,:,:,0]*Cab[:,:,:,2]) + A[:,:,:,0] * (Ba[:,:,:,1]*Ca[:,:,:,2] + Ba[:,:,:,2]*Ca[:,:,:,1])

    Gxy = Gxy.to(torch.complex64)
    Gxz = Gxz.to(torch.complex64)
    Gyz = Gyz.to(torch.complex64)

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)

    Gxy = Gxy * areas
    Gxz = Gxz * areas
    Gyz = Gyz * areas

    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points, board)

    Exy = Fxy + Gxy@H
    Exz = Fxz + Gxz@H
    Eyz = Fyz + Gyz@H

    return Exy, Exz, Eyz

def BEM_gorkov_analytical(activations,points,scatterer=None,board=TRANSDUCERS,H=None,E=None):
    if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
    
    if E is None:
        E = compute_E(scatterer,points,board,H=H)

    Ex, Ey, Ez = BEM_forward_model_grad(points,scatterer,board,H=H)

    p = E@activations
    px = Ex@activations
    py = Ey@activations
    pz = Ez@activations

    
    K1 = Constants.V / (4*Constants.p_0*Constants.c_0**2)
    K2 = 3*Constants.V / (4*(2*Constants.f**2 * Constants.p_0))

    U = K1 * torch.abs(p)**2 - K2*(torch.abs(px)**2 + torch.abs(py)**2 + torch.abs(pz)**2)

    return U

def BEM_force_analytical(activations,points,scatterer=None,board=TRANSDUCERS,H=None,E=None,return_components=False, axis=None):
    E = compute_E(scatterer, points, board)
    Exx, Eyy, Ezz = BEM_forward_model_second_derivative_unmixed(points,scatterer,board)
    Exy, Exz, Eyz = BEM_forward_model_second_derivative_mixed(points,scatterer,board)
    Ex, Ey, Ez = BEM_forward_model_grad(points,scatterer,board)

    p = E@activations
    px = Ex@activations
    py = Ey@activations
    pz = Ez@activations
    pxx = Exx@activations
    pyy = Eyy@activations
    pzz = Ezz@activations
    pxy = Exy@activations
    pxz = Exz@activations
    pyz = Eyz@activations

    K1 = Constants.V / (4*Constants.p_0*Constants.c_0**2)
    K2 = 3*Constants.V / (4*(2*Constants.f**2 * Constants.p_0))

    P = torch.abs(p) 
    Px = torch.abs(px) 
    Py = torch.abs(py) 
    Pz = torch.abs(pz) 
    
    Pxx = torch.abs(pxx) 
    Pyy = torch.abs(pyy) 
    Pzz = torch.abs(pzz) 
    Pxy = torch.abs(pxy) 
    Pxz = torch.abs(pxz) 
    Pyz = torch.abs(pyz)


    single_sum = 2*K2*(Pz+Py+Pz)
    force_x = -1 * (2*P * (K1 * Px - K2*(Pxz+Pxy+Pxx)) - Px*single_sum)
    force_y = -1 * (2*P * (K1 * Py - K2*(Pyz+Pyy+Pxy)) - Py*single_sum)
    force_z = -1 * (2*P * (K1 * Pz - K2*(Pzz+Pyz+Pxz)) - Pz*single_sum)

    force = torch.cat([force_x, force_y, force_z],2)

    if axis is not None:
        return force[:,:,axis]

    if return_components:
        return force_x, force_y, force_z
    else:
        return force

    



if __name__ == "__main__":
    from acoustools.Solvers import wgs_batch
    from acoustools.Gorkov import gorkov_fin_diff, gorkov_analytical
    from acoustools.Utilities import add_lev_sig
    from acoustools.Visualiser import Visualise


    paths = ["Media/flat-lam1.stl","Media/flat-lam1.stl"]
    # scatterer = load_scatterer(path)
    board = TRANSDUCERS
    scatterer = load_multiple_scatterers(paths,board,dys=[-0.06,0.06,0],dxs=[0,0,0.06],rotxs=[-90,90,0],rotys=[0,0,90])
    origin = (0,0,-0.06)
    normal = (1,0,0)

    N=4
    B = 1
    points = create_points(N,B)


    E = compute_E(scatterer, points, board)
    Exx, Eyy, Ezz = BEM_forward_model_second_derivative_unmixed(points,scatterer,board)
    Exy, Exz, Eyz = BEM_forward_model_second_derivative_mixed(points,scatterer,board)
    Ex, Ey, Ez = BEM_forward_model_grad(points,scatterer,board)
    
    F = forward_model_batched(points, board)
    _,_,x = wgs_batch(E, torch.ones(N,1).to(device)+0j,200)
    # _,_,xF = wgs_batch(F, torch.ones(N,1).to(device)+0j,200)
    x = add_lev_sig(x)
    # xF = add_lev_sig(xF)
  
    force = BEM_force_analytical(x,points,scatterer,board)
    print(force)



    # A = torch.tensor((0,-0.07, 0.07))
    # B = torch.tensor((0,0.07, 0.07))
    # C = torch.tensor((0,-0.07, -0.07))
    # res = (100,100)

    # Visualise(A,B,C,x,colour_functions=[BEM_force_analytical],points=points,res=res,colour_function_args=[{"axis":1, "scatterer":scatterer,"board":TRANSDUCERS}])


    exit()

    U_ag = gorkov_fin_diff(x,points,prop_function=propagate_BEM,prop_fun_args={"scatterer":scatterer,"board":board},K1=K1, K2=K2)
    print(U_ag)

   
    print(U.squeeze_())

    UF = gorkov_analytical(xF,points,board).squeeze_()

    print(U / UF)

    x = add_lev_sig(x)
    xF = add_lev_sig(xF)
    p_BEM = E@x
    p_f = F@xF

    print(torch.abs(p_BEM).squeeze_() / torch.abs(p_f).squeeze_())
    # print(K1 * torch.abs(p)**2)


    # vedo.show(scatterer)   


    '''
    E = compute_E(scatterer,points,board,print_lines=True) #E=F+GH

    from Solvers import wgs
    _, _,x1 = wgs(E[0,:],torch.ones(N,1).to(device)+0j,200)
    _, _,x2 = wgs(E[1,:],torch.ones(N,1).to(device)+0j,200)
    x = torch.stack([x1,x2])
    

    # print(E)
    # print(x.shape)
    print(propagate_BEM_pressure(x,points,E=E))
    
    '''



