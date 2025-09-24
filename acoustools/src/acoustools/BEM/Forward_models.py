
import torch
from torch import Tensor

from vedo import Mesh
from typing import Literal
import hashlib
import pickle

import acoustools.Constants as Constants

from acoustools.Utilities import device, DTYPE, forward_model_batched, TOP_BOARD
from acoustools.Mesh import get_normals_as_points, board_name, get_centres_as_points

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
    norms= norms.real
    vecs = y.real-x.real

 
    distance = torch.sqrt(torch.sum((vecs)**2,dim=3))

    norms = norms.expand(B,N,-1,-1)

    
    # norm_norms = torch.norm(norms,2,dim=3) # === 1
    # vec_norms = torch.norm(vecs,2,dim=3) # === distance?
    # print(vec_norms == distance)
    angles = (torch.sum(norms*vecs,3) / (distance))

    del norms, vecs
    torch.cuda.empty_cache()

    A = -1 * greens(y,x,distance=distance)
    B = (1j*Constants.k - 1/(distance))
    
    del distance
    # torch.cuda.empty_cache()

    partial_greens = A*B*angles
    
    if not return_components:
        del A,B,angles
    torch.cuda.empty_cache()

    
    partial_greens[partial_greens.isnan()] = 1


    if return_components:
        return partial_greens, A,B,angles
    

    
    return partial_greens 

def greens(y:Tensor,x:Tensor, k:float=Constants.k, distance=None):
    '''
    Computes greens function for a source at y and a point at x\n
    :param y: source location
    :param x: point location
    :param k: wavenumber
    :param distance: precomputed distances from y->x
    :returns greens function:
    '''
    if distance is None:
        vecs = y.real-x.real
        distance = torch.sqrt(torch.sum((vecs)**2,dim=3)) 
    green = -1 * torch.exp(1j*k*distance) / (4*Constants.pi*distance)

    return green

def compute_G(points: Tensor, scatterer: Mesh, k:float=Constants.k, alphas:float|Tensor=1, betas:float|Tensor = 0) -> Tensor:
    '''
    Computes G in the BEM model\n
    :param points: The points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :return G: `torch.Tensor` of G
    '''
    torch.cuda.empty_cache()
    areas = torch.Tensor(scatterer.celldata["Area"]).to(device).real
    B = points.shape[0]
    N = points.shape[2]
    M = areas.shape[0]
    areas = areas.expand((B,N,-1))

    #Compute the partial derivative of Green's Function

    #Firstly compute the distances from mesh points -> control points
    centres = torch.tensor(scatterer.cell_centers().points).to(device).real #Uses centre points as position of mesh
    centres = centres.expand((B,N,-1,-1))
    
    # print(points.shape)
    # p = torch.reshape(points,(B,N,3))
    p = torch.permute(points,(0,2,1)).real
    p = torch.unsqueeze(p,2).expand((-1,-1,M,-1))

    #Compute cosine of angle between mesh normal and point
    # scatterer.compute_normals()
    # norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = get_normals_as_points(scatterer,permute_to_points=False).real.expand((B,N,-1,-1))

    # centres_p = get_centres_as_points(scatterer)
    
    
    partial_greens = compute_green_derivative(centres,p,norms, B,N,M)
    
    
    if ((type(betas) in [int, float]) and betas != 0) or (type(betas) is Tensor and (betas != 0).any()):  #Either β non 0 and type(β) is number or β is Tensor and any elemenets non 0
        green = greens(centres, p, k=k) * 1j * k * betas
        partial_greens += green
    
    G = areas * partial_greens

    if ((type(alphas) in [int, float]) and alphas != 1) or (type(alphas) is Tensor and (alphas != 1).any()):
        #Does this need to be in A too?
        if type(alphas) is Tensor:
            alphas = alphas.unsqueeze(2)
            alphas = alphas.expand(B, N, M)
        vecs = p - centres
        distance = torch.norm(vecs, p=2, dim=3)
        angle = torch.sum(vecs * norms, dim=3) / distance
        angle = angle.real
        print(G.shape, angle.shape)
        G[angle>0] = G[angle>0] * alphas
    
    return G

 
def compute_A(scatterer: Mesh, k:float=Constants.k, betas = 0) -> Tensor:
    '''
    Computes A for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements

    :return A: A tensor
    '''

    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)

    centres = torch.tensor(scatterer.cell_centers().points).to(DTYPE).to(device)
    m = centres
    M = m.shape[0]
    m = m.expand((M,M,3))

    m_prime = m.clone()
    m_prime = m_prime.permute((1,0,2))

    # norms = torch.tensor(scatterer.cell_normals).to(device)
    norms = get_normals_as_points(scatterer,permute_to_points=False)

    partial_greens = compute_green_derivative(m.unsqueeze_(0),m_prime.unsqueeze_(0),norms,1,M,M)
    if betas != 0:  
        green = greens(m, m_prime, k=k) * 1j * k * betas
        partial_greens += green


    # areas = areas.unsqueeze(0).T.expand((-1,M)).unsqueeze(0)
    A = partial_greens * areas * -1
    eye = torch.eye(M).to(bool)
    A[:,eye] = 0.5
    return A.to(DTYPE)

 
def compute_bs(scatterer: Mesh, board:Tensor, p_ref=Constants.P_ref, norms:Tensor|None=None) -> Tensor:
    '''
    Computes B for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :return B: B tensor
    '''
    centres = torch.tensor(scatterer.cell_centers().points).to(DTYPE).to(device).T.unsqueeze_(0)
    bs = forward_model_batched(centres,board, p_ref=p_ref,norms=norms)
    return bs

 
def compute_H(scatterer: Mesh, board:Tensor ,use_LU:bool=True, use_OLS:bool = False, p_ref = Constants.P_ref, norms:Tensor|None=None, k:float=Constants.k, betas = 0) -> Tensor:
    '''
    Computes H for the BEM model \n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param use_LU: if True computes H with LU decomposition, otherwise solves using standard linear inversion
    :param use_OLS: if True computes H with OLS, otherwise solves using standard linear inversion
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements

    :return H: H
    '''
    
    A = compute_A(scatterer, k=k, betas=betas)
    bs = compute_bs(scatterer,board,p_ref=p_ref,norms=norms)
    if use_LU:
        LU, pivots = torch.linalg.lu_factor(A)
        H = torch.linalg.lu_solve(LU, pivots, bs)
    elif use_OLS:
       
        H = torch.linalg.lstsq(A,bs).solution    
    else:
         H = torch.linalg.solve(A,bs)

    return H



def get_cache_or_compute_H(scatterer:Mesh,board,use_cache_H:bool=True, path:str="Media", 
                           print_lines:bool=False, cache_name:str|None=None, p_ref = Constants.P_ref, 
                           norms:Tensor|None=None, method=Literal['OLS','LU', 'INV'], k:float=Constants.k, betas = 0) -> Tensor:
    '''
    Get H using cache system. Expects a folder named BEMCache in `path`\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param  board: Transducers to use 
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it
    :param path: path to folder containing `BEMCache/ `
    :param print_lines: if true prints messages detaling progress
    :param method: Method to use to compute H: One of OLS (Least Squares), LU. (LU decomposition). If INV (or anything else) will use `torch.linalg.solve`
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :return H: H tensor
    '''

    use_OLS=False
    use_LU = False
    
    if method == "OLS":
        use_OLS = True
    elif method == "LU":
        use_LU = True
    
    
    if use_cache_H:
        
        if cache_name is None:
            cache_name = scatterer.filename+"--"+ board_name(board) + '--' + str(p_ref)
            cache_name = hashlib.md5(cache_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  cache_name + ".bin"
        # print(f_name)

        try:
            if print_lines: print("Trying to load H at", f_name ,"...")
            H = pickle.load(open(f_name,"rb")).to(device).to(DTYPE)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H...")
            H = compute_H(scatterer,board,use_LU=use_LU,use_OLS=use_OLS,norms=norms, k=k, betas=betas)
            f = open(f_name,"wb")
            pickle.dump(H,f)
            f.close()
    else:
        if print_lines: print("Computing H...")
        H = compute_H(scatterer,board, p_ref=p_ref,norms=norms,use_LU=use_LU,use_OLS=use_OLS, k=k, betas=betas)

    return H

def compute_E(scatterer:Mesh, points:Tensor, board:Tensor|None=None, use_cache_H:bool=True, print_lines:bool=False,
               H:Tensor|None=None,path:str="Media", return_components:bool=False, p_ref = Constants.P_ref, norms:Tensor|None=None, H_method=None, k:float=Constants.k, betas = 0, alphas:float|Tensor=1) -> Tensor:
    '''
    Computes E in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use, if `None` then `acoustools.Utilities.TOP_BOARD` is used
    :param use_cache_H_grad: If true uses the cache system, otherwise computes H and does not save it
    :param print_lines: if true prints messages detaling progress
    :param H: Precomputed H - if None H will be compute
    :param path: path to folder containing `BEMCache/`
    :param return_components: if true will return the subparts used to compute, F,G,H
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements

    :return E: Propagation matrix for BEM E

    ```Python
    from acoustools.Mesh import load_scatterer
    from acoustools.BEM import compute_E, propagate_BEM_pressure, compute_H
    from acoustools.Utilities import create_points, TOP_BOARD
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise

    import torch

    path = "../../BEMMedia"
    scatterer = load_scatterer(path+"/Sphere-lam2.stl",dy=-0.06,dz=-0.08)
    
    p = create_points(N=1,B=1,y=0,x=0,z=0)
    
    H = compute_H(scatterer, TOP_BOARD)
    E = compute_E(scatterer, p, TOP_BOARD,path=path,H=H)
    x = wgs(p,board=TOP_BOARD,A=E)
    
    A = torch.tensor((-0.12,0, 0.12))
    B = torch.tensor((0.12,0, 0.12))
    C = torch.tensor((-0.12,0, -0.12))

    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],
                colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H}],
                vmax=8621, show=True,res=[256,256])
    ```
    
    '''
    if board is None:
        board = TOP_BOARD

    if norms is None: #Transducer Norms
        norms = (torch.zeros_like(board) + torch.tensor([0,0,1], device=device)) * torch.sign(board[:,2].real).unsqueeze(1).to(DTYPE)

    if print_lines: print("H...")
    
    if H is None:
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines,p_ref=p_ref,norms=norms, method=H_method, k=k, betas=betas).to(DTYPE)
        
    if print_lines: print("G...")
    G = compute_G(points, scatterer, k=k, betas=betas,alphas=alphas).to(DTYPE)
    
    if print_lines: print("F...")
    F = forward_model_batched(points,board,p_ref=p_ref,norms=norms).to(DTYPE)
    
    if print_lines: print("E...")

    E = F+G@H

    torch.cuda.empty_cache()
    if return_components:
        return E.to(DTYPE), F.to(DTYPE), G.to(DTYPE), H.to(DTYPE)
    return E.to(DTYPE)


