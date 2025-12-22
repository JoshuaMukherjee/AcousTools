import torch
from torch import Tensor

from vedo import Mesh

import hashlib, pickle

from acoustools.Utilities import device, DTYPE, forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed
from acoustools.BEM.Forward_models import compute_bs, compute_A, get_cache_or_compute_H
from acoustools.Mesh import get_centres_as_points, get_normals_as_points, board_name
import acoustools.Constants as Constants

from acoustools.BEM.Gradients.E_Gradient import get_G_partial


 
def grad_H(points: Tensor, scatterer: Mesh, transducers: Tensor, return_components:bool = False, 
           path:str='', H:Tensor=None, use_cache_H:bool=True) ->tuple[Tensor,Tensor, Tensor] | tuple[Tensor,Tensor, Tensor, Tensor,Tensor, Tensor, Tensor]:
    '''
    @private
    Computes the gradient of H wrt scatterer centres\n
    Ignores `points` - for compatability with other gradient functions, takes centres of the scatterers
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param transducers: Transducers to use 
    :param return_components: if true will return the subparts used to compute the derivative
    :return grad_H: The gradient of the H matrix wrt the position of the mesh
    '''
    print("Implementation not tested H grad - probably do correct")
    if H is None:
        H = get_cache_or_compute_H(scatterer, transducers, use_cache_H, path)
    
    
    # centres = torch.tensor(scatterer.cell_centers().points).to(device).T.unsqueeze_(0)
    centres = get_centres_as_points(scatterer)


    M = centres.shape[2]

    B = compute_bs(scatterer,transducers)
    A = compute_A(scatterer)
    A_inv = torch.inverse(A).to(DTYPE)

    
    Bx, By, Bz = forward_model_grad(centres, transducers)
    Bx = Bx.to(DTYPE) 
    By = By.to(DTYPE)
    Bz = Bz.to(DTYPE)


    Ax, Ay, Az =  get_G_partial(centres,scatterer,transducers)
    # Ax *= -1
    # Ay *= -1
    # Az *= -1
    
    Ax = (-1* Ax)
    Ay = (-1* Ay)
    Az = (-1* Az)


    
    eye = torch.eye(M).to(bool)
    Ax[:,eye] = 0
    Ay[:,eye] = 0
    Az[:,eye] = 0
    
    # A_inv_x = (-1*A_inv @ Ax @ A_inv).to(DTYPE)
    # A_inv_y = (-1*A_inv @ Ay @ A_inv).to(DTYPE)
    # A_inv_z = (-1*A_inv @ Az @ A_inv).to(DTYPE)

    # Hx_old = (A_inv_x@B) + (A_inv@Bx)
    # Hy_old = (A_inv_y@B) + (A_inv@By)
    # Hz_old = (A_inv_z@B) + (A_inv@Bz)


    Hx = A_inv @ (Bx - Ax @ H)
    Hy = A_inv @ (By - Ay @ H)
    Hz = A_inv @ (Bz - Az @ H)


    Hx = Hx.to(DTYPE)
    Hy = Hy.to(DTYPE)
    Hz = Hz.to(DTYPE)

    if return_components:
        return Hx, Hy, Hz, A, A_inv, Ax, Ay, Az
    else:
        return Hx, Hy, Hz

 
def grad_2_H(points: Tensor, scatterer: Mesh, transducers: Tensor, A:Tensor|None = None, 
             A_inv:Tensor|None = None, Ax:Tensor|None = None, Ay:Tensor|None = None, Az:Tensor|None = None) -> Tensor:
    '''
    @private
    Computes the second derivative of H wrt scatterer centres\n
    Ignores `points` - for compatability with other gradient functions, takes centres of the scatterers
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param transducers: Transducers to use 
    :param A: The result of a call to `compute_A`
    :param A_inv: The inverse of `A`
    :param Ax: The gradient of A wrt the x position of scatterer centres
    :param Ay: The gradient of A wrt the y position of scatterer centres
    :param Az: The gradient of A wrt the z position of scatterer centres
    :return Haa: second order unmixed gradient of H wrt scatterer positions
    '''
    print("Implementation not tested H grad - probably do correct")
    centres = get_centres_as_points(scatterer)
    M = centres.shape[2]

    B = compute_bs(scatterer,transducers)

    Fx, Fy, Fz = forward_model_grad(centres, transducers)
    Fx = Fx.to(DTYPE)
    Fy = Fy.to(DTYPE)
    Fz = Fz.to(DTYPE)
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
        Ax = Ax.to(DTYPE)
        Ay = Ay.to(DTYPE)
        Az = Az.to(DTYPE)
    Aa = torch.stack([Ax,Ay,Az],dim=3)

    
    A_inv_x = (-1*A_inv @ Ax @ A_inv).to(DTYPE)
    A_inv_y = (-1*A_inv @ Ay @ A_inv).to(DTYPE)
    A_inv_z = (-1*A_inv @ Az @ A_inv).to(DTYPE)


    A_inv_a = torch.stack([A_inv_x,A_inv_y,A_inv_z],dim=3)

    m = centres.permute(0,2,1)
    m = m.expand((M,M,3))

    m_prime = m.clone()
    m_prime = m_prime.permute((1,0,2))

    vecs = m - m_prime
    vecs = vecs.unsqueeze(0)
    

    # norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = get_normals_as_points(scatterer,permute_to_points=False)
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
    Gaa = Gaa.to(DTYPE)

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

    Fa = Fa.to(DTYPE)
    Faa = Faa.to(DTYPE)

    Gaa = Gaa.permute(0,3,2,1)
    Aa = Aa.permute(0,3,2,1)
    Aa = Aa.to(DTYPE)

    X1 = A_inv_a @ Fa + A_inv @ Faa
    X2 = (A_inv @ (Aa @ A_inv @ Aa - Gaa)@A_inv) @ F
    X3 = A_inv_a@Fa


    Haa = X1 + X2 + X3
    
    return Haa

 
def get_cache_or_compute_H_2_gradients(scatterer:Mesh,board:Tensor,use_cache_H_grad:bool=True, path:str="Media", print_lines:bool=False) -> Tensor:
    '''
    @private
    Get second derivatives of H using cache system. Expects a folder named BEMCache in `path`\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it
    :param path: path to folder containing `BEMCache/ `
    :param print_lines: if true prints messages detaling progress
    :return: second derivatives of H
    '''
    print("Implementation not tested H grad - probably do correct")
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

 
def get_cache_or_compute_H_gradients(scatterer,board,use_cache_H_grad=True, path="Media", print_lines=False) -> tuple[Tensor, Tensor, Tensor]:
    '''
    @private
    Get derivatives of H using cache system. Expects a folder named BEMCache in `path`\\
    :param scatterer: The mesh used (as a `vedo` `mesh` object)\\
    :param board: Transducers to use \\
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it\\
    :param path: path to folder containing BEMCache/ \\
    :param print_lines: if true prints messages detaling progress\\
    Returns derivatives of H
    '''
    print("Implementation not tested for H grad - probably do correct")
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
            Hx, Hy, Hz = grad_H(None, transducers=board, **{"scatterer":scatterer }, path=path)
            f = open(f_name,"wb")
            pickle.dump((Hx, Hy, Hz),f)
            f.close()
    else:
        if print_lines: print("Computing H Grad...")
        Hx, Hy, Hz = grad_H(None, transducers=board, **{"scatterer":scatterer }, path=path)

    return Hx, Hy, Hz

