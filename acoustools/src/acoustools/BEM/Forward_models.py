
import torch
from torch import Tensor

from vedo import Mesh
from typing import Literal
import hashlib
import pickle


import acoustools.Constants as Constants

from acoustools.Utilities import device, DTYPE, forward_model_batched, TOP_BOARD, create_points
from acoustools.Mesh import get_normals_as_points, get_centres_as_points, get_areas, get_centre_of_mass_as_points

from acoustools.Utilities import forward_model_grad





def compute_green_derivative(y:Tensor,x:Tensor,norms:Tensor,B:int,N:int,M:int, return_components:bool=False, 
                             a:Tensor=None, c:Tensor=None, k=Constants.k, smooth_distance:float=0) -> Tensor:
    '''
    Computes the derivative of greens function \n
    :param y: y in greens function - location of the source of sound
    :param x: x in greens function - location of the point to be propagated to
    :param norms: norms to y 
    :param B: Batch dimension
    :param N: size of x
    :param M: size of y
    :param return_components: if true will return the subparts used to compute the derivative \n
    :param k: Wavenumber to use
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :return: returns the partial derivative of greeens fucntion wrt y
    '''
    norms= norms.real

    vecs = y.real-x.real

 
    distance = torch.sqrt(torch.sum((vecs)**2,dim=3))
    
    if a is None: #Were computing with a OR we have no a to begin with
        if len(vecs.shape) > 4: #Vecs isnt expanded - we must never have had an a
            norms = norms.unsqueeze(4).expand(B,N,-1,-1,1)
        else: #vecs included a 
            norms = norms.expand(B,N,-1,-1)
    else:
        norms = norms.expand(B,N,-1,-1)

    
    # norm_norms = torch.norm(norms,2,dim=3) # === 1x
    # vec_norms = torch.norm(vecs,2,dim=3) # === distance?
    # print(vec_norms == distance)
    angles = (torch.sum(norms*vecs,3) / (distance))

    # del norms, vecs
    torch.cuda.empty_cache()

    # distance = distance + smooth_distance
    distance = distance.clamp_min(smooth_distance)


    A = 1 * greens(y,x,distance=distance,k=k)
    ik_d = (1j*k - 1/(distance))
    
    del distance
    # torch.cuda.empty_cache()

    partial_greens = A*ik_d*angles
    
    # if not return_components:
    #     del A,B,angles
    torch.cuda.empty_cache()

    

    if a is not None:
        n_a = a.shape[2]
        # a = a.permute(0,2,1)
        a = a.unsqueeze(1).unsqueeze(2)
        a = a.expand(B,N,M,3,n_a).clone()
        y = y.unsqueeze(4).expand(B,N,M,3,n_a)
        g_mod =  torch.sum(c*compute_green_derivative(y, a, norms, B, N, M,k=k),dim=3) #Allow for multiple a's
        partial_greens += g_mod
    
    
    partial_greens[partial_greens.isnan()] = 0
    if return_components:
        return partial_greens, A,ik_d,angles
    

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
    green = torch.exp(1j*k*distance) / (4*Constants.pi*distance)

    return green

def compute_G(points: Tensor, scatterer: Mesh, k:float=Constants.k, alphas:float|Tensor=1, betas:float|Tensor = 0, a:Tensor=None, c:Tensor=None, smooth_distance:float=0) -> Tensor:
    '''
    Computes G in the BEM model\n
    :param points: The points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements. If Tensor, should have shape [B,M] where M is the mesh size, B is the batch size and will normally be 1
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
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
    partial_greens = compute_green_derivative(centres,p,norms, B,N,M, a=a, c=c, k=k,smooth_distance=smooth_distance )
    
    if ((type(betas) in [int, float]) and betas != 0) or (type(betas) is Tensor and (betas != 0).any()):  #Either β non 0 and type(β) is number or β is Tensor and any elemenets non 0
        green = greens(centres, p, k=k) * 1j * k * betas
        partial_greens += green
    
    G = areas * partial_greens


    if ((type(alphas) in [int, float]) and alphas != 1) or (type(alphas) is Tensor and (alphas != 1).any()):
        #Does this need to be in A too?
        if type(alphas) is Tensor:
            alphas = alphas.unsqueeze(1)
            alphas = alphas.expand(B, N, M)
        vecs = p - centres
        angle = torch.sum(vecs * norms, dim=3) #Technically not the cosine of the angle - would need to /distance but as we only care about the sign then it doesnt matter
        angle = angle.real
        if type(alphas) is Tensor:
            G[angle>0] = G[angle>0] * alphas[angle>0]
        else:
            G[angle>0] = G[angle>0] * alphas
    
    return G



def augment_A_CHIEF(A:Tensor, internal_points:Tensor, CHIEF_mode:Literal['square', 'rect'] = 'square', centres:Tensor=None, norms:Tensor=None, areas:Tensor=None, scatterer:Mesh=None, k:float=Constants.k):
    '''
    Augments an A matrix with CHIEF BEM equations\n
    
    :param A: A matrix
    :param internal_points:  The internal points to use for CHIEF based BEM
    :param CHIEF_mode: Mode for CHIEF -> augment A to be either rectangular (only appended to one axis) or square (appended to both with a 0 block in the corner)
    :param centres: mesh centres
    :param norms: mesh normals
    :param areas: mesh aeras
    :param scatterer: mesh
    :param k: wavenumber
    '''
     # if internal_points is not None: print(internal_points.shape)

    if internal_points is not None and (internal_points.shape[1] == 3 and internal_points.shape[2] != 3):
        internal_points = internal_points.permute(0,2,1)

    if areas is None: areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
    if centres is None: centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    if norms is None: norms = get_normals_as_points(scatterer, permute_to_points=False)



    P = internal_points.shape[1]
    M = centres.shape[0]

    m_int = centres.unsqueeze(0).unsqueeze(1)
    int_p = internal_points.unsqueeze(2)
    # G_block = greens(m_int,int_p, k=k) 
    
    int_norms = norms.unsqueeze(1)
    G_block = -compute_green_derivative(m_int,int_p, int_norms, 1, P, M, k=k)
    G_block = G_block * areas[None,None,:] 
    
    G_block_t = G_block.mT
    zero_block = torch.zeros((1, P, P), device=device, dtype=DTYPE)

    
    # return torch.cat((A, G_block), dim=1)
    if CHIEF_mode.lower() == 'square':
        A_aux = torch.cat((A, G_block_t), dim=2)
        
        GtZ = torch.cat((G_block, zero_block), dim=2)
        A = torch.cat((A_aux, GtZ), dim=1)
    elif CHIEF_mode.lower() == 'rect':
        A = torch.cat([A, G_block], dim= 1)
    else:
        raise RuntimeError(f"Invalid CHIEF Mode {CHIEF_mode}, should be on of [square, rect]")

    return A




def compute_A(scatterer: Mesh, k:float=Constants.k, alphas:float|Tensor = 1, betas:float|Tensor = 0, a:Tensor=None, c:Tensor=None, internal_points:Tensor=None, smooth_distance:float=0, CHIEF_mode:Literal['square', 'rect'] = 'square', h:float=None, BM_alpha:complex=None, BM_mode:str='fd') -> Tensor:
    '''
    Computes A for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param internal_points: The internal points to use for CHIEF based BEM
    :param CHIEF_mode: Mode for CHIEF -> augment A to be either rectangular (only appended to one axis) or square (appended to both with a 0 block in the corner)
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM
    :param BM_mode: Mode to use for Burton-Miller BEM - should be fd for finite differences
    

    :return A: A tensor
    '''

    
    
    areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
    centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    norms = get_normals_as_points(scatterer, permute_to_points=False)

    M = centres.shape[0]
    # if internal_points is not None:
        # M = M + internal_points.shape[2]

    if h is not None: #We want to do fin-diff BM
        if BM_alpha is None: #We are in the grad step
            centres = centres - h * norms.squeeze(0)
        else:
            if BM_mode != 'analytical':
                Aminus = compute_A(scatterer=scatterer, k=k, betas=betas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, CHIEF_mode=CHIEF_mode, h=h, BM_alpha=None)
                Aplus = compute_A(scatterer=scatterer, k=k, betas=betas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, CHIEF_mode=CHIEF_mode, h=-h, BM_alpha=None)

            else: #This will need to move as h is not required for analytical
                A_grad = torch.stack(__get_G_partial(centres.unsqueeze(0).permute(0,2,1), scatterer, None, k=k), dim=1)
                # A_grad = A_grad.permute(0,1,3,2)

                n = norms.permute(0,2,1).unsqueeze(3)
                A_norm = -1 * torch.sum(A_grad * n , dim=1)
            
    m = centres.expand((M, M, 3))
    m_prime = centres.unsqueeze(1).expand((M, M, 3))


    partial_greens = compute_green_derivative(m.unsqueeze_(0), m_prime.unsqueeze_(0), norms, 1, M, M, a=a, c=c, k=k,smooth_distance=smooth_distance)
    

    if ((type(betas) in [int, float]) and betas != 0) or (isinstance(betas, Tensor) and (betas != 0).any()):
        green = greens(m, m_prime, k=k) * 1j * k * betas
        partial_greens += green

    A = -partial_greens * areas
    A[:, torch.eye(M, dtype=torch.bool, device=device)] = 0.5
    

    if internal_points is not None: #CHIEF

        A = augment_A_CHIEF(A, internal_points, CHIEF_mode, centres, norms, areas, scatterer,k=k)
     

    if BM_alpha is not None: #Burton-Miller F.D

        if BM_mode == 'analytical':
            A_grad = A_norm
        
        else:
            A_grad = (Aplus - Aminus) / (2*h)


        A_grad[:, torch.eye(A_grad.shape[2], dtype=torch.bool, device=device)] = 0.5     
        A = A - BM_alpha * A_grad

        A[:, torch.eye(A_grad.shape[2], dtype=torch.bool, device=device)] = 0.5     


    if ((type(alphas) in [int, float]) and alphas != 1) or (type(alphas) is Tensor and (alphas != 1).any()):
        if type(alphas) is Tensor:
            alphas = alphas.unsqueeze(1)
            alphas = alphas.expand(1, M, M)
        vecs = m_prime - m
        angle = torch.sum(vecs * norms, dim=3) #Technically not the cosine of the angle - would need to /distance but as we only care about the sign then it doesnt matter
        angle = angle.real
        if type(alphas) is Tensor:
            A[angle>0] = A[angle>0] * alphas[angle>0]
        else:
            A[angle>0] = A[angle>0] * alphas
    
    return A.to(DTYPE)

 
def compute_bs(scatterer: Mesh, board:Tensor, p_ref=Constants.P_ref, norms:Tensor|None=None, 
               a:Tensor=None, c:Tensor=None, k=Constants.k, internal_points:Tensor=None, h:float=None, 
               BM_alpha:complex=None, BM_mode:str='analytical', transducer_radius = Constants.radius) -> Tensor:
    '''
    Computes B for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param internal_points: The internal points to use for CHIEF based BEM
    :param CHIEF_mode: Mode for CHIEF -> augment A to be either rectangular (only appended to one axis) or square (appended to both with a 0 block in the corner)
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM
    :param BM_mode: Mode to use for Burton-Miller BEM - should be analytical
    :param k: wavenumber
    :return B: B tensor
    '''

    if norms is None:
        norms = (torch.zeros_like(board) + torch.tensor([0,0,1], device=device)) * torch.sign(board[:,2].real).unsqueeze(1).to(DTYPE)


    centres = torch.tensor(scatterer.cell_centers().points).to(DTYPE).to(device).T.unsqueeze_(0)
    if h is not None: #Burton-Miller F.D
        mesh_norms = get_normals_as_points(scatterer, permute_to_points=True)
        if BM_alpha is None: #We are in the grad step
            centres = centres - h * mesh_norms.squeeze(0)
        elif BM_mode != 'analytical':
            bs_grad = compute_bs(scatterer=scatterer, board=board, p_ref=p_ref, norms=norms, a=a, c=c, k=k, internal_points=internal_points,h=h, BM_alpha=None, transducer_radius=transducer_radius)

    bs = forward_model_batched(centres,board, p_ref=p_ref,norms=norms,k=k, transducer_radius=transducer_radius) 


    if internal_points is not None: #CHIEF
        F_int = forward_model_batched(internal_points.permute(0,2,1), board, p_ref=p_ref,norms=norms,k=k, transducer_radius=transducer_radius)
        bs = torch.cat([bs, F_int], dim=1)
    
    if a is not None: #Modified Greens function
        f_mod = torch.sum(forward_model_batched(a,board, p_ref=p_ref,norms=norms, k=k, transducer_radius=transducer_radius), dim=1, keepdim=True)
        bs += c * f_mod
    
    
    if BM_alpha is not None: #Burton-Miller
        
        if BM_mode == 'analytical': 
            bs_a_grad = torch.stack(forward_model_grad(centres, board, p_ref=p_ref, k=k, transducer_radius=transducer_radius), dim=1)
            bs_norm_grad = torch.sum(bs_a_grad * mesh_norms.unsqueeze(3), dim=1)


            if internal_points is not None: #CHIEF
                # int_bs_grad = torch.stack(forward_model_grad(internal_points.permute(0,2,1), board, p_ref=p_ref, k=k, transducer_radius=transducer_radius), dim=1)
                p_n = internal_points.shape[1]
                M = board.shape[0]
                # int_bs_grad = torch.zeros_like(int_bs_grad)[:,0,:]
                int_bs_grad = torch.zeros((1,p_n,M))

                bs_norm_grad = torch.cat([bs_norm_grad,int_bs_grad], dim=1)


            
            bs = bs - BM_alpha * bs_norm_grad
        else:
            bs_grad = (bs-bs_grad)/h
            bs = bs - BM_alpha * (bs-bs_grad)/h


    return bs   

 
def compute_H(scatterer: Mesh, board:Tensor ,use_LU:bool=True, use_OLS:bool = False, p_ref = Constants.P_ref, norms:Tensor|None=None, k:float=Constants.k, alphas:float|Tensor = 1, betas:float|Tensor = 0, 
              a:Tensor=None, c:Tensor=None, internal_points:Tensor=None, smooth_distance:float=0, 
              return_components:bool=False, CHIEF_mode:Literal['square', 'rect'] = 'square', h:float=None, BM_alpha:complex=None, transducer_radius = Constants.radius, A:Tensor|None=None, bs:Tensor|None=None) -> Tensor:
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
    :param internal_points: The internal points to use for CHIEF based BEM
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param CHIEF_mode: Mode for CHIEF -> augment A to be either rectangular (only appended to one axis) or square (appended to both with a 0 block in the corner)
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM
    :return H: H
    '''

    # internal_points = None

    centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    M = centres.shape[0]

    if internal_points is not None and (internal_points.shape[1] == 3 and internal_points.shape[2] != 3):
            internal_points = internal_points.permute(0,2,1)

    
    # if internal_points is not None: print(internal_points.shape)
    


    if A is None: A = compute_A(scatterer, alphas=alphas, betas=betas, a=a, c=c, k=k,internal_points=internal_points, smooth_distance=smooth_distance, CHIEF_mode=CHIEF_mode, h=h, BM_alpha=BM_alpha)
    if bs is None: bs = compute_bs(scatterer,board,p_ref=p_ref,norms=norms,a=a,c=c, k=k,internal_points=internal_points, h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius)


    if use_LU:
        LU, pivots = torch.linalg.lu_factor(A)
        H = torch.linalg.lu_solve(LU, pivots, bs)
    elif use_OLS: 
       
        H = torch.linalg.lstsq(A,bs, rcond=1e-6).solution    
    else:
         H = torch.linalg.solve(A,bs)
    
    # H = H / (1-eta*1j)

    # exit()
    H = H[:,:M,: ]
    
    # print(torch.linalg.eig(A))

    # exit()

    # print((k*H, A@H, bs), H/bs)
    # exit()

    if return_components: return H,A,bs
    return H



def get_cache_or_compute_H(scatterer:Mesh,board,use_cache_H:bool=True, path:str="Media", 
                           print_lines:bool=False, cache_name:str|None=None, p_ref = Constants.P_ref, 
                           norms:Tensor|None=None, method:Literal['OLS','LU', 'INV']='LU', k:float=Constants.k,alphas:float|Tensor = 1, betas:float|Tensor = 0, 
                           a:Tensor=None, c:Tensor=None, internal_points:Tensor=None, smooth_distance:float=0, 
                           CHIEF_mode:Literal['square', 'rect'] = 'square', h:float=None, BM_alpha:complex=None, transducer_radius=Constants.radius) -> Tensor:
    '''
    Get H using cache system. Expects a folder named BEMCache in `path`\n

    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param use_LU: if True computes H with LU decomposition, otherwise solves using standard linear inversion
    :param use_OLS: if True computes H with OLS, otherwise solves using standard linear inversion
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param internal_points: The internal points to use for CHIEF based BEM
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param CHIEF_mode: Mode for CHIEF -> augment A to be either rectangular (only appended to one axis) or square (appended to both with a 0 block in the corner)
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM
    :param use_cache_H: If true uses the cache system, otherwise computes H and does not save it
    :param path: path to folder containing `BEMCache/ `
    :param print_lines: if true prints messages detaling progress
    :param method: Method to use to compute H: One of OLS (Least Squares), LU. (LU decomposition). If INV (or anything else) will use `torch.linalg.solve`
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
            ps = locals()
            ps.__delitem__('scatterer')
            cache_name = scatterer.filename+"--" + str(ps)
            cache_name = hashlib.md5(cache_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  cache_name + ".bin"
        # print(f_name)

        try:
            if print_lines: print("Trying to load H at", f_name ,"...")
            H = pickle.load(open(f_name,"rb")).to(device).to(DTYPE)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H...")
            H = compute_H(scatterer,board,use_LU=use_LU,use_OLS=use_OLS,norms=norms, k=k, alphas=alphas, betas=betas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, CHIEF_mode=CHIEF_mode,h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius)
            try:
                f = open(f_name,"wb")
            except FileNotFoundError as e:
                print(e)
                raise FileNotFoundError("AcousTools BEM expects a directory named BEMCache inside of `path' in order to use the cache and this was not found. Check this directory exists")
            pickle.dump(H,f)
            f.close()
    else:
        if print_lines: print("Computing H...")
        H = compute_H(scatterer,board, p_ref=p_ref,norms=norms,use_LU=use_LU,use_OLS=use_OLS, k=k, alphas=alphas, betas=betas, a=a, c=c, internal_points=internal_points, smooth_distance=smooth_distance, CHIEF_mode=CHIEF_mode, h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius)

    return H

def compute_E(scatterer:Mesh, points:Tensor, board:Tensor|None=None, use_cache_H:bool=True, print_lines:bool=False,
               H:Tensor|None=None,path:str="Media", return_components:bool=False, p_ref = Constants.P_ref, norms:Tensor|None=None, H_method:Literal['OLS','LU', 'INV']=None, 
               k:float=Constants.k, betas:float|Tensor = 0, alphas:float|Tensor=1, a:Tensor=None, c:Tensor=None, internal_points:Tensor=None, smooth_distance:float=0,  CHIEF_mode:Literal['square', 'rect'] = 'square', h:float=None, BM_alpha:complex=None, transducer_radius=Constants.radius) -> Tensor:
    '''
    Computes E in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param use_LU: if True computes H with LU decomposition, otherwise solves using standard linear inversion
    :param use_OLS: if True computes H with OLS, otherwise solves using standard linear inversion
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param internal_points: The internal points to use for CHIEF based BEM
    :param a: position to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param c: constant to use for a modified-green's function based BEM formulation (see (18) in `Eliminating the fictitious frequency problem in BEM solutions of the external Helmholtz equation` for more details)
    :param smooth_distance: amount to add to distances to avoid explosion over small values
    :param CHIEF_mode: Mode for CHIEF -> augment A to be either rectangular (only appended to one axis) or square (appended to both with a 0 block in the corner)
    :param h: finite difference step for Burton-Miller BEM
    :param BM_alpha: constant alpha to use in Burton-Miller BEM
    :param use_cache_H: If true uses the cache system, otherwise computes H and does not save it
    :param path: path to folder containing `BEMCache/ `
    :param print_lines: if true prints messages detaling progress
    :param method: Method to use to compute H: One of OLS (Least Squares), LU. (LU decomposition). If INV (or anything else) will use `torch.linalg.solve`

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
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines,p_ref=p_ref,
                                   norms=norms, method=H_method, k=k, betas=betas, a=a, c=c, internal_points=internal_points,
                                     smooth_distance=smooth_distance, CHIEF_mode=CHIEF_mode, h=h, BM_alpha=BM_alpha, transducer_radius=transducer_radius, alphas=alphas).to(DTYPE)
        
    if print_lines: print("G...")
    G = compute_G(points, scatterer, k=k, betas=betas,alphas=alphas, smooth_distance=smooth_distance).to(DTYPE)
    
    if print_lines: print("F...")
    F = forward_model_batched(points,board,p_ref=p_ref,norms=norms, k=k, transducer_radius=transducer_radius).to(DTYPE)  
    # if a is not None:
    #     F += c * forward_model_batched(a,board, p_ref=p_ref,norms=norms)
    
    if print_lines: print("E...")

    E = F+G@H


    torch.cuda.empty_cache()
    if return_components:
        return E.to(DTYPE), F.to(DTYPE), G.to(DTYPE), H.to(DTYPE)
    return E.to(DTYPE)





def __get_G_partial(points:Tensor, scatterer:Mesh, board:Tensor|None=None, return_components:bool=False, k=Constants.k) -> tuple[Tensor, Tensor, Tensor]:
    '''
    @private
    here so it can be used in Burton-Miller BEM above - not ideal for it to be here so this should be refactored at some point
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

    diff = (points - centres)
    diff_square = diff**2
    distances = torch.sqrt(torch.sum(diff_square, 1))
    distances_expanded = distances.unsqueeze(1)#.expand((1,3,N,M))
    distances_expanded_square = distances_expanded**2
    distances_expanded_cube = distances_expanded**3

    # G  =  e^(ikd) / 4pi d
    G = areas * torch.exp(1j * k * distances_expanded) / (4*3.1415*distances_expanded)

    #Ga =  [i*da * e^{ikd} * (kd+i) / 4pi d^2]

    #d = distance
    #da = -(at - a)^2 / d
    da = diff / distances_expanded
    kd = k * distances_expanded
    phase = torch.exp(1j*kd)
    Ga =  areas * ( (1j*da*phase * (kd + 1j))/ (4*3.1415*distances_expanded_square))

    #P = (ik - 1/d)
    P = (1j*k - 1/distances_expanded)
    #Pa = da / d^2 = (diff / d^2) /d
    Pa = diff / distances_expanded_cube

    #C = (diff \cdot normals) / distances

    nx = normals[:,0]
    ny = normals[:,1]
    nz = normals[:,2]

    dx = diff[:,0,:]
    dy = diff[:,1,:]
    dz = diff[:,2,:]

    n_dot_d = nx*dx + ny*dy + nz*dz

    C = (n_dot_d) / distances


    distance_square = distances**2


    Cx = 1/distance_square * (nx * distances - (n_dot_d * dx) / distances)
    Cy = 1/distance_square * (ny * distances - (n_dot_d * dy) / distances)
    Cz = 1/distance_square * (nz * distances - (n_dot_d * dz) / distances)

    Cx.unsqueeze_(1)
    Cy.unsqueeze_(1)
    Cz.unsqueeze_(1)

    Ca = torch.cat([Cx, Cy, Cz],axis=1)

    grad_G = Ga*P*C + G*P*Ca + G*Pa*C

    grad_G =  -grad_G.to(DTYPE)

    if return_components:
        return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:], G,P,C,Ga,Pa, Ca
    
    return grad_G[:,0,:], grad_G[:,1,:], grad_G[:,2,:]
