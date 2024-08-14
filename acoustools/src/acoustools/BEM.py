'''
Module to deal with the scattering of sound off of objects.
See
High-speed acoustic holography with arbitrary scattering objects: https://doi.org/10.1126/sciadv.abn7614
BEM notes: https://www.personal.reading.ac.uk/~sms03snc/fe_bem_notes_sncw.pdf 
'''

import torch
import pickle
from vedo import Mesh
from torch import Tensor

import matplotlib.pyplot as plt

from acoustools.Utilities import device, TOP_BOARD, TRANSDUCERS, forward_model_batched, create_points, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, DTYPE
import acoustools.Constants as Constants
from acoustools.Mesh import scatterer_file_name, load_scatterer, load_multiple_scatterers, get_centres_as_points, board_name, get_areas, get_normals_as_points

import hashlib

 
def compute_green_derivative(y:Tensor,x:Tensor,norms:Tensor,B:int,N:int,M:int, return_components:bool=False) -> Tensor:
    '''
    Computes the derivative of greens function \n
    :param y: y in greens function - location of the source of sound
    :param x: x in greens function - location of the point to be propagated to
    :param norms: norms to y 
    :param B: Batch dimension
    :param N: size of x
    :param M: size of y
    :param return_components: if true will return the subparts used to compute the derivative \n
    :return: returns the partial derivative of greeens fucntion wrt y
    '''
    distance = torch.sqrt(torch.sum((x - y)**2,dim=3))

    vecs = y-x
    norms = norms.expand(B,N,-1,-1)

    
    # norm_norms = torch.norm(norms,2,dim=3) # === 1
    vec_norms = torch.norm(vecs,2,dim=3)
    angles = (torch.sum(norms*vecs,3) / (vec_norms)).to(DTYPE)

    A = ((torch.e**(1j*Constants.k*distance))/(4*torch.pi*distance)).to(DTYPE)
    B = (1j*Constants.k - 1/(distance)).to(DTYPE)
    partial_greens = A*B*angles
    partial_greens[partial_greens.isnan()] = 1

    if return_components:
        return partial_greens, A,B,angles
    
    return partial_greens

 
def compute_G(points: Tensor, scatterer: Mesh) -> Tensor:
    '''
    Computes G in the BEM model\n
    :param points: The points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :return G: `torch.Tensor` of G
    '''
    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)
    B = points.shape[0]
    N = points.shape[2]
    M = areas.shape[0]
    areas = areas.expand((B,N,-1))

    #Compute the partial derivative of Green's Function

    #Firstly compute the distances from mesh points -> control points
    centres = torch.tensor(scatterer.cell_centers).to(device).to(DTYPE) #Uses centre points as position of mesh
    centres = centres.expand((B,N,-1,-1))
    
    # print(points.shape)
    # p = torch.reshape(points,(B,N,3))
    p = torch.permute(points,(0,2,1))
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    #Compute cosine of angle between mesh normal and point
    scatterer.compute_normals()
    # norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = get_normals_as_points(scatterer,permute_to_points=False)
  
    partial_greens = compute_green_derivative(centres,p,norms, B,N,M)
    
    G = areas * partial_greens
    return G

 
def compute_A(scatterer: Mesh) -> Tensor:
    '''
    Computes A for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :return A: A tensor
    '''

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)

    centres = torch.tensor(scatterer.cell_centers).to(device)
    m = centres
    M = m.shape[0]
    m = m.expand((M,M,3))

    m_prime = m.clone()
    m_prime = m_prime.permute((1,0,2))

    # norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = get_normals_as_points(scatterer,permute_to_points=False)

    green = compute_green_derivative(m.unsqueeze_(0),m_prime.unsqueeze_(0),norms,1,M,M)
    # areas = areas.unsqueeze(0).T.expand((-1,M)).unsqueeze(0)
    A = green * areas * -1
    eye = torch.eye(M).to(bool)
    A[:,eye] = 0.5

    return A.to(DTYPE)

 
def compute_bs(scatterer: Mesh, board:Tensor) -> Tensor:
    '''
    Computes B for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :return B: B tensor
    '''
    centres = torch.tensor(scatterer.cell_centers).to(device).T.unsqueeze_(0)
    bs = forward_model_batched(centres,board)
    return bs.to(DTYPE)

 
def compute_H(scatterer: Mesh, board:Tensor ,use_LU:bool=True, use_OLS:bool = False) -> Tensor:
    '''
    Computes H for the BEM model \n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param use_LU: if True computes H with LU decomposition, otherwise solves using standard linear inversion
    :return H: H
    '''
    
    A = compute_A(scatterer)
    bs = compute_bs(scatterer,board)
    if use_LU:
        LU, pivots = torch.linalg.lu_factor(A)
        H = torch.linalg.lu_solve(LU, pivots, bs)
    elif use_OLS:
       
        H = torch.linalg.lstsq(A,bs).solution    
    else:
         H = torch.linalg.solve(A,bs)

    return H

 
def grad_H(points: Tensor, scatterer: Mesh, transducers: Tensor, return_components:bool = False) ->tuple[Tensor,Tensor, Tensor] | tuple[Tensor,Tensor, Tensor, Tensor,Tensor, Tensor, Tensor]:
    '''
    Computes the gradient of H wrt scatterer centres\n
    Ignores `points` - for compatability with other gradient functions, takes centres of the scatterers
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param transducers: Transducers to use 
    :param return_components: if true will return the subparts used to compute the derivative
    :return grad_H: The gradient of the H matrix wrt the position of the mesh
    '''
    centres = torch.tensor(scatterer.cell_centers).to(device).T.unsqueeze_(0)

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

    
    A_inv_x = (-1*A_inv @ Ax @ A_inv).to(DTYPE)
    A_inv_y = (-1*A_inv @ Ay @ A_inv).to(DTYPE)
    A_inv_z = (-1*A_inv @ Az @ A_inv).to(DTYPE)

    Hx = (A_inv_x@B) + (A_inv@Bx)
    Hy = (A_inv_y@B) + (A_inv@By)
    Hz = (A_inv_z@B) + (A_inv@By)

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
    Get second derivatives of H using cache system. Expects a folder named BEMCache in `path`\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it
    :param path: path to folder containing `BEMCache/ `
    :param print_lines: if true prints messages detaling progress
    :return: second derivatives of H
    '''
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
    Get derivatives of H using cache system. Expects a folder named BEMCache in `path`\\
    :param scatterer: The mesh used (as a `vedo` `mesh` object)\\
    :param board: Transducers to use \\
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it\\
    :param path: path to folder containing BEMCache/ \\
    :param print_lines: if true prints messages detaling progress\\
    Returns derivatives of H
    '''
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

 
def get_cache_or_compute_H(scatterer:Mesh,board,use_cache_H:bool=True, path:str="Media", 
                           print_lines:bool=False, cache_name:str|None=None,use_LU:bool=True) -> Tensor:
    '''
    Get H using cache system. Expects a folder named BEMCache in `path`\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param  board: Transducers to use 
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it
    :param path: path to folder containing `BEMCache/ `
    :param print_lines: if true prints messages detaling progress
    :param use_LU: If true use LU decomopsition to solve for H
    :return H: H tensor
    '''

    if use_cache_H:
        
        if cache_name is None:
            cache_name = scatterer.filename+"--"+ board_name(board)
            cache_name = hashlib.md5(cache_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  cache_name + ".bin"
        # print(f_name)

        try:
            if print_lines: print("Trying to load H at", f_name ,"...")
            H = pickle.load(open(f_name,"rb")).to(device)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H...")
            H = compute_H(scatterer,board,use_LU=use_LU)
            f = open(f_name,"wb")
            pickle.dump(H,f)
            f.close()
    else:
        if print_lines: print("Computing H...")
        H = compute_H(scatterer,board)

    return H
 
def compute_E(scatterer:Mesh, points:Tensor, board:Tensor|None=None, use_cache_H:bool=True, print_lines:bool=False,
               H:Tensor|None=None,path:str="Media", return_components:bool=False) -> Tensor:
    '''
    Computes E in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then `acoustools.Utilities.TOP_BOARD` is used
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it
    :param print_lines: if true prints messages detaling progress
    :param H: Precomputed H - if None H will be compute
    :param path: path to folder containing `BEMCache/`
    :param return_components: if true will return the subparts used to compute, F,G,H
    :return E: Propagation matrix for BEM E

    ```Python
            from acoustools.Mesh import load_scatterer
            from acoustools.BEM import compute_E, propagate_BEM_pressure, compute_H
            from acoustools.Utilities import create_points, TOP_BOARD
            from acoustools.Solvers import wgs
            from acoustools.Visualiser import Visualise

            import torch, vedo

            path = "../../BEMMedia"
            scatterer = load_scatterer(path+"/Sphere-lam2.stl",dy=-0.06,dz=-0.08)
            
            N=1
            B=1
            p = create_points(N,B,y=0,x=0,z=0)
            
            H = compute_H(scatterer, TOP_BOARD)
            E, F, G, H = compute_E(scatterer, p, TOP_BOARD,path=path,use_cache_H=False,return_components=True,H=H)
            x = wgs(p,board=TOP_BOARD,A=E)
            
            A = torch.tensor((-0.12,0, 0.12))
            B = torch.tensor((0.12,0, 0.12))
            C = torch.tensor((-0.12,0, -0.12))
            normal = (0,1,0)
            origin = (0,0,0)

            line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

            Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H}],vmax=8621, show=True,res=[256,256])
    ```
    
    '''
    if board is None:
        board = TOP_BOARD

    if print_lines: print("H...")
    
    if H is None:
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines).to(DTYPE)
        
    if print_lines: print("G...")
    G = compute_G(points, scatterer).to(DTYPE)
    
    if print_lines: print("F...")
    F = forward_model_batched(points,board).to(DTYPE)
    
    if print_lines: print("E...")

    E = F+G@H

    if return_components:
        return E.to(DTYPE), F.to(DTYPE), G.to(DTYPE), H.to(DTYPE)
    return E.to(DTYPE)

 
def propagate_BEM(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:Tensor|None=None,
                  E:Tensor|None=None,path:str="Media", use_cache_H: bool=True,print_lines:bool=False) ->Tensor:
    '''
    Propagates transducer phases to points using BEM\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then uses `acoustools.Utilities.TOP_BOARD` 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :param path: path to folder containing `BEMCache/ `
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    :return pressure: complex pressure at points
    '''
    if board is None:
        board = TOP_BOARD

    if E is None:
        if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
        E = compute_E(scatterer,points,board,H=H, path=path,use_cache_H=use_cache_H,print_lines=print_lines)
    
    out = E@activations
    return out

 
def propagate_BEM_pressure(activations:Tensor,points:Tensor,scatterer:Mesh|None=None,board:Tensor|None=None,H:
                           Tensor|None=None,E:Tensor|None=None, path:str="Media",use_cache_H:bool=True, print_lines:bool=False) -> Tensor:
    '''
    Propagates transducer phases to points using BEM and returns absolute value of complex pressure\n
    Equivalent to `torch.abs(propagate_BEM(activations,points,scatterer,board,H,E,path))` \n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed 
    :param path: path to folder containing `BEMCache/ `
    
    :param use_cache_H: If True uses the cache system to load and save the H matrix. Default `True`
    :param print_lines: if true prints messages detaling progress
    
    :return pressure: real pressure at points
    '''
    if board is None:
        board = TOP_BOARD

    point_activations = propagate_BEM(activations,points,scatterer,board,H,E,path,use_cache_H=use_cache_H,print_lines=print_lines)
    pressures =  torch.abs(point_activations)
    return pressures

 
def get_G_partial(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False) -> tuple[Tensor, Tensor, Tensor]:
    '''
    Computes gradient of the G matrix in BEM \n
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` will use `acoustools.Utilities.TRANSDUCERS`
    :param return_components: if true will return the subparts used to compute
    :return: Gradient of the G matrix in BEM
    '''
    #Bk1. Page 273
    if board is None:
        board = TRANSDUCERS
    areas = get_areas(scatterer)
    centres = get_centres_as_points(scatterer)

    N = points.shape[2]
    M = centres.shape[2]

    points = points.unsqueeze(3).expand(-1,-1,-1,M)
    centres = centres.unsqueeze(2).expand(-1,-1,N,-1)

    vecs = points - centres #Centres -> Points
    vecs = vecs.to(DTYPE)
    distances = torch.sum(vecs**2)
    norms = get_normals_as_points(scatterer).to(DTYPE).unsqueeze(2).expand(-1,-1,N,-1)

    vec_norm = torch.norm(vecs,2,dim=1)
    angle = torch.einsum('ijkh,ijkh->ikh', vecs, norms).unsqueeze(1) / vec_norm
    angle_grad = -1*norms / vec_norm
    phase = torch.exp(1j * Constants.k * distances)

    grad_G = areas * (-1 * phase / (4*torch.pi*distances**3) * (vecs / distances * angle * (Constants.k**2 * distances**2 + 2j*Constants.k*distances - 2) + distances * angle_grad * (1-1j*Constants.k*distances)))
    grad_G = grad_G.to(DTYPE)
    
    return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:]

 
def BEM_forward_model_grad(points:Tensor, scatterer:Mesh, transducers:Tensor|Mesh=None, use_cache_H:bool=True, 
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
    if board is None:
        board = TRANSDUCERS

    B = points.shape[0]
    if H is None:
        H = get_cache_or_compute_H(scatterer,transducers,use_cache_H, path, print_lines)
    
    Fx, Fy, Fz  = forward_model_grad(points, transducers)
    Gx, Gy, Gz = get_G_partial(points, scatterer, transducers)


    Gx[Gx.isnan()] = 0
    Gy[Gy.isnan()] = 0
    Gz[Gz.isnan()] = 0

    Fx = Fx.to(DTYPE)
    Fy = Fy.to(DTYPE)
    Fz = Fz.to(DTYPE)

    H = H.expand(B, -1, -1).to(DTYPE)


    Ex = Fx + Gx@H
    Ey = Fy + Gy@H
    Ez = Fz + Gz@H


    if return_components:
        return Ex.to(DTYPE), Ey.to(DTYPE), Ez.to(DTYPE), Fx, Fy, Fz, Gx, Gy, Gz, H
    else:
        return Ex.to(DTYPE), Ey.to(DTYPE), Ez.to(DTYPE)


def BEM_gorkov_analytical(activations:Tensor,points:Tensor,scatterer:Mesh|None|str=None,
                          board:Tensor|None=None,H:Tensor|None=None,E:Tensor|None=None,
                          **params) -> Tensor:
    '''
    Returns Gor'kov potential computed analytically from the BEM model\n
    :param activations: Transducer hologram
    :param points: Points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object) or string of path to mesh
    :param board: Transducers to use 
    :param H: Precomputed H - if None H will be computed
    :param E: Precomputed E - if None E will be computed
    :return: Gor'kov potential at point U
    '''
    if board is None:
        board = TRANSDUCERS
    if type(scatterer) == str:
            scatterer = load_scatterer(scatterer)
    
    path = params['path']
    
    if E is None:
        E = compute_E(scatterer,points,board,H=H,path=path)

    Ex, Ey, Ez = BEM_forward_model_grad(points,scatterer,board,H=H,path=path)

    p = E@activations
    px = Ex@activations
    py = Ey@activations
    pz = Ez@activations

    
    K1 = Constants.V / (4*Constants.p_0*Constants.c_0**2)
    K2 = 3*Constants.V / (4*(2*Constants.f**2 * Constants.p_0))

    U = K1 * torch.abs(p)**2 - K2*(torch.abs(px)**2 + torch.abs(py)**2 + torch.abs(pz)**2)

    return U
