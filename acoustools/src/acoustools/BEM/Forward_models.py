
import torch
from torch import Tensor

from vedo import Mesh
from typing import Literal
import hashlib
import pickle

import acoustools.Constants as Constants

from acoustools.Utilities import device, DTYPE, forward_model_batched, TOP_BOARD
from acoustools.Mesh import get_normals_as_points, board_name, get_centres_as_points, get_areas, get_barycentric_points, get_cell_verticies

from acoustools.Utilities import forward_model_grad





def compute_green_derivative(y:Tensor,x:Tensor,norms:Tensor,B:int,N:int,M:int, return_components:bool=False, a=None, c=None, k=Constants.k) -> Tensor:
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

def compute_G(points: Tensor, scatterer: Mesh, k:float=Constants.k, alphas:float|Tensor=1, betas:float|Tensor = 0, a=None, c=None) -> Tensor:
    '''
    Computes G in the BEM model\n
    :param points: The points to propagate to
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param k: wavenumber
    :param alphas: Absorbance of each element, can be Tensor for element-wise attribution or a number for all elements. If Tensor, should have shape [B,M] where M is the mesh size, B is the batch size and will normally be 1
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
    partial_greens = compute_green_derivative(centres,p,norms, B,N,M, a=a, c=c, k=k )
    
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






def compute_An(scatterer: Mesh, k:float=Constants.k, eps_scale:float = 0.2, eps:float=0) -> Tensor:

    def An_Kernel(m, m_prime, n, n_prime, eps=0, k=Constants.k):

        direction = m - m_prime
        distance = torch.sqrt(torch.sum(direction**2, dim=2))

        distance = torch.sqrt(distance**2 + eps**2).real
        # distance = torch.clamp(distance, min=5e-6)
        # print(m)
        # print()
        # print(m_prime)

        # exit()

        R_hat = direction / distance.unsqueeze(2)

        nm_dot_nn = torch.sum(n_prime * n, dim=2)
        Rnm_dot_nm = torch.sum(R_hat * n_prime, dim=2)
        Rnm_dot_nn = torch.sum(R_hat * n, dim=2)

        common = torch.exp(1j * k * distance) / (4 * Constants.pi * distance**3)

        kernel = common * (
            (1 - 1j * k * distance)* (3 * Rnm_dot_nm * Rnm_dot_nn - nm_dot_nn)
            - (k**2 * distance**2) * Rnm_dot_nm * Rnm_dot_nn
        )
    
        return kernel


    
    areas = torch.Tensor(scatterer.celldata["Area"]).to(device)



    areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
    centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    norms = get_normals_as_points(scatterer, permute_to_points=False).squeeze(0)

    if eps == 0:
        eps = eps_scale * torch.sqrt(areas).unsqueeze(1)
        eps = eps.unsqueeze(-1)
    

    M = centres.shape[0]



    b_centres, w = get_barycentric_points(scatterer)
    
    bm = b_centres.squeeze(0).permute(1,0,2).unsqueeze(0).expand(M,M,3,7)
    bm_prime = centres.unsqueeze(1).unsqueeze(3).expand(M,M,3,7)




    bn_m = norms.unsqueeze(0).unsqueeze(3)
    bn_n = norms.unsqueeze(1).unsqueeze(3)

    kernel = An_Kernel(bm, bm_prime, bn_m, bn_n, eps, k)
    w = w.unsqueeze(0).unsqueeze(1)
    kernel = kernel * w
    kernel = torch.sum(kernel,dim=2)

    


    # Analytic self-term
    eye = torch.eye(M, dtype=torch.bool, device=device)

    kernel[eye] = - (k**2 / (4 * Constants.pi))

    kernel = kernel * areas

    
    return kernel


def compute_auxiliary_AB(scatterer: Mesh, k=Constants.k, eps = Constants.wavelength/4):

    def get_phi_dphi(source, target, norms):
        
        # target = target.unsqueeze(1)
        
        vecs = target - source #Other way around?
        distance = torch.sqrt(torch.sum(vecs**2, dim=2))

        mu = torch.sum(norms * vecs, dim=2) / distance
        
        Green = torch.exp(1j * k * distance) / (4 * 3.1415 * distance)
        grad_G = Green/distance * (1j * k *distance - 1)
        grad_2_G = Green * (-k**2 - 2j*k/distance + 2/distance**2)
        dGreen = grad_G * mu

        Dipole = -1 * dGreen
        dDipole = - (grad_2_G * mu**2 + grad_G/distance * (1-mu**2))

        denom = (dGreen**2 +dDipole*Green)
        alpha =  dDipole / denom #In theory these are always the same? -> Will leave computing evrything for now?
        beta = -1*dGreen / denom #Same as above

    

        target = target.unsqueeze(1)
        phi_vecs = target - source
        phi_dist = torch.sqrt(torch.sum(phi_vecs**2,dim=2))
        
        phi_G = torch.exp(1j * k * phi_dist) / (4 * 3.1415 * phi_dist)
        
        phi_grad_G = phi_G/phi_dist * (1j * k *phi_dist - 1)
        phi_mu = torch.sum(norms * phi_vecs, dim=2) / phi_dist
        phi_D = -1 * phi_grad_G*phi_mu
        
        phi = alpha * phi_G + beta * phi_D

        phi_upsilon =  torch.sum(norms.permute(1,0,2) * phi_vecs, dim=2) / phi_dist

        
        phi_rho = torch.sum(norms.permute(1,0,2) * norms, dim=2)
        phi_d_2_G = phi_G * (-k**2 - 2j*k/phi_dist + 2/phi_dist**2) 

        dphi = -alpha * phi_grad_G * phi_upsilon + beta * (phi_d_2_G * phi_upsilon * phi_rho + phi_grad_G/phi_dist * (phi_rho - phi_upsilon*phi_mu))

        return phi, dphi, alpha, beta
    
    areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
    centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    norms = get_normals_as_points(scatterer, permute_to_points=False) #Might need to permute 0 & 1

    auxiliary_points = centres - eps*norms

    phi, dphi, alpha, beta = get_phi_dphi(auxiliary_points, centres, norms)
    import matplotlib.pyplot as plt

    # Assume you already have:
    # alpha: [1, M], beta: [1, M]
    # phi: [M, M], dphi: [M, M]
    # auxiliary_points: [1, M, 3], centres: [M, 3], norms: [1, M, 3]
    # eps, k already defined

    M = centres.shape[0]

    # --- 1. Check alpha/beta broadcasting ---
    alpha_exp = alpha.expand(M, M)
    beta_exp = beta.expand(M, M)
    print("alpha_exp shape:", alpha_exp.shape, "beta_exp shape:", beta_exp.shape)

    # --- 2. Check vector subtraction and distances ---
    vecs = centres.unsqueeze(1) - auxiliary_points  # Should be target - source
    distances = torch.sqrt(torch.sum(vecs**2, dim=2))
    print("vecs shape:", vecs.shape, "distances shape:", distances.shape)
    print("distances min/max:", distances.real.min().item(), distances.real.max().item())

    # --- 3. Compute simple Green ---
    Green = torch.exp(1j * k * distances) / (4 * 3.1415 * distances)
    plt.imshow(Green.abs().cpu(), cmap='viridis')
    plt.title("Green magnitude")
    plt.colorbar()
    plt.show()

    # --- 4. Check self-term / diagonal ---
    diag_vals = torch.diag(Green)
    plt.plot(diag_vals.abs().cpu())
    plt.title("Diagonal values of Green")
    plt.show()
    print("Diagonal min/max:", diag_vals.real.min().item(), diag_vals.real.max().item())

    # --- 5. Check row/column repetition (broadcasting artifacts) ---
    row_variation = torch.std(Green, dim=1)
    col_variation = torch.std(Green, dim=0)
    plt.plot(row_variation.cpu(), label='row std')
    plt.plot(col_variation.cpu(), label='col std')
    plt.title("Row/Column std of Green (should not be ~0)")
    plt.legend()
    plt.show()
    print("Row std min/max:", row_variation.real.min().item(), row_variation.real.max().item())
    print("Col std min/max:", col_variation.real.min().item(), col_variation.real.max().item())

    # --- 6. Check alpha*phi + beta*dphi pattern ---
    phi_test = alpha_exp * phi + beta_exp * dphi
    plt.imshow(phi_test.abs().cpu(), cmap='magma')
    plt.title("phi pattern (alpha*phi + beta*dphi)")
    plt.colorbar()
    plt.show()

    centres = centres.unsqueeze(1)

    
    ab_vecs = centres - centres.permute(1,0,2)
    ab_dist = torch.sqrt(torch.sum(ab_vecs**2, dim=2))
    ab_G = torch.exp(1j * k * ab_dist) / (4 * 3.1415 * ab_dist)
    ab_gradG = ab_G/ab_dist * (1j * k *ab_dist - 1)

    ab_mu = torch.sum(norms * ab_vecs, dim=2) / ab_dist

    ab_dG = ab_gradG * ab_mu

    A_aux = areas.unsqueeze(1) * (ab_G * dphi - ab_dG*phi)

    cell_verts = get_cell_verticies(scatterer)
    #When m == m' -> Goes to 0 -> goes to NaN, so we want to replace the diagonal with the sum of the effect at three little triangles from the centre
    #So we want Mx3? -> M sources contributing to 3 verts each -> But different ones for each -> Then sum over 3 and replace diagonal (times area etc)
    #So have expand centres to Mx3 -> (centres, centres, centres) and do the same thing as above but with the two parameters being Mx3 centres and some Mx3 set of points for each triangle
    #But what are the points per triangle - multiply verts by (2/3, 1/6, 1/6)
    bary_w = torch.tensor([[1/6, 1/6, 2/3], [1/6, 2/3, 1/6], [2/3, 1/6, 1/6]]) + 0j

    bary_points = cell_verts @ bary_w #1x3xMxP => p is the number of points per subtriangle

    # This section makes a Mx3M matrix -> Not sure i want that but might be useful for comparision later?
    # print(bary_points.shape)
    # a,b,c = torch.tensor_split(bary_points, 3, 3)
    # bary_points = torch.cat([a,b,c], dim=2).squeeze(3).permute(0,2,1)
    # bary_vecs = bary_points - centres
    # bary_distances = torch.sqrt(torch.sum(bary_vecs**2, dim=2)) #This is (3*M) x M ->  Remember to reshape later
    # #Now -> X is the centroids and y is the barycentric centres -> M x 3M -> use that to find the function on each of y -> then reshape & sum ->

    #bary_centre is the targets, centres will be the source
    bary_points = bary_points.permute(2, 0,1,3) # 1x3xMxP -> Mx1x3xP
    centres_bary = centres.unsqueeze(3)


    norms_bary = norms.unsqueeze(3).permute(1,0,2,3)
    bary_aux_source = centres_bary + eps*norms_bary
    
    bary_vecs = bary_points - bary_aux_source
    bary_distances = torch.sqrt(torch.sum(bary_vecs**2, dim=2))
    bary_vec_norms = bary_vecs /bary_distances.unsqueeze(3)

    bary_norms = norms.permute(1,0,2).unsqueeze(3)
    
    bary_mu = torch.sum(bary_norms * bary_vec_norms, dim=2)
    bary_upsilon = bary_mu
    bary_rho = torch.sum(bary_norms * bary_norms, dim=2)

    bary_Green = torch.exp(1j * k * bary_distances) / (4 * 3.1415 * bary_distances)
    bary_grad_G = bary_Green/bary_distances * (1j * k *bary_distances - 1)
    bary_grad_2_G = bary_Green * (-k**2 - 2j*k/bary_distances + 2/bary_distances**2)

    bary_alpha = alpha[0,:].unsqueeze(1).unsqueeze(2)
    bary_beta = beta[0,:].unsqueeze(1).unsqueeze(2)

    bary_phi = bary_alpha * bary_Green - bary_beta * bary_grad_G * bary_mu
    bary_dphi = -bary_alpha * bary_grad_G * bary_upsilon + bary_beta * (bary_grad_2_G * bary_upsilon * bary_rho + bary_grad_G/bary_distances * (bary_rho - bary_upsilon*bary_mu))
    # bary_phi_diag = bary_phi[torch.eye(bary_dphi.shape[0], dtype=bool),:].unsqueeze(1)   
    # bary_dphi_diag = bary_dphi[torch.eye(bary_dphi.shape[0], dtype=bool),:].unsqueeze(1)   
    bary_phi_diag = bary_phi
    bary_dphi_diag = bary_dphi

    bary_phi_diag = bary_phi_diag - 1 #Subtract the contribiution at centre

    barry_centre_vecs = bary_points - centres.unsqueeze(3)
    barry_centre_distance = torch.sqrt(torch.sum(barry_centre_vecs**2, dim=2))
    
    bary_centre_Green = torch.exp(1j * k * barry_centre_distance) / (4 * 3.1415 * barry_centre_distance)
    bary_centre_grad_Green = bary_centre_Green/barry_centre_distance * (1j * k *barry_centre_distance - 1)

    bary_centre_mu = torch.sum(barry_centre_vecs * norms_bary) / barry_centre_distance
    bary_centre_partial_Green = -1 * bary_centre_grad_Green * bary_centre_mu

    g = bary_centre_Green * bary_dphi_diag - bary_centre_partial_Green*bary_phi_diag
    g = 1/3 * torch.sum(g, dim=2) * areas.unsqueeze(1) 

    A_aux[torch.eye(A_aux.shape[0], dtype = (bool))] = g.squeeze(1) #+ 1/2
    # A_aux[torch.eye(A_aux.shape[0], dtype = (bool))] = 1/2


    B_aux = torch.sum(A_aux, dim=1)

    

    # exit()
    return A_aux, B_aux


def compute_A(scatterer: Mesh, k:float=Constants.k, betas:float|Tensor = 0, a=None, c=None, BM_c=None, internal_points=None) -> Tensor:
    '''
    Computes A for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param k: wavenumber
    :param betas: Ratio of impedances of medium and scattering material for each element, can be Tensor for element-wise attribution or a number for all elements
    :param internal_points: The internal points to use for CHIEF based BEM

    :return A: A tensor
    '''

    

    areas = torch.tensor(scatterer.celldata["Area"], dtype=DTYPE, device=device)
    centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    norms = get_normals_as_points(scatterer, permute_to_points=False)

    M = centres.shape[0]
    m = centres.expand((M, M, 3))
    m_prime = centres.unsqueeze(1).expand((M, M, 3))

    partial_greens = compute_green_derivative(m.unsqueeze_(0), m_prime.unsqueeze_(0), norms, 1, M, M, a=a, c=c, k=k)
    

    if ((type(betas) in [int, float]) and betas != 0) or (isinstance(betas, Tensor) and (betas != 0).any()):
        green = greens(m, m_prime, k=k) * 1j * k * betas
        partial_greens += green

    # Core double-layer operator
    A = -partial_greens * areas
    A[:, torch.eye(M, dtype=torch.bool, device=device)] = 0.5

    if internal_points is not None:

        P = internal_points.shape[1]

        m_int = centres.unsqueeze(0).unsqueeze(1)
        int_p = internal_points.unsqueeze(2)
        # G_block = greens(m_int,int_p, k=k) 
        
        int_norms = norms.unsqueeze(1)
        G_block = -compute_green_derivative(m_int,int_p, int_norms, 1, P, M, k=k)
        G_block = G_block * areas[None,None,:] 
        
        G_block_t = G_block.mT
        zero_block = torch.zeros((1, P, P), device=device, dtype=DTYPE)

        
        A_aux = torch.cat((A, G_block_t), dim=2)
        GtZ = torch.cat((G_block, zero_block), dim=2)
        A = torch.cat((A_aux, GtZ), dim=1)

        # A = torch.cat([A, G_block], dim=1)

        # import matplotlib.pyplot as plt
        # plt.matshow(A.real[0,:])
        # plt.colorbar()
        # plt.show()
        # exit()



    # Add BM hypersingular correction if supplied
    if BM_c is not None:

        vec = m - m_prime
        distance = torch.sqrt(torch.sum(vec**2, dim=3))

        G = torch.exp(1j* k * distance)/(4*3.1415 * distance)  * areas
        G[:,torch.eye(G.shape[-1], dtype=bool)] = areas / (4*3.1415) * (torch.log(areas/4) - 1) + (1j * areas * k) / (4*3.1415) 
        A = A - 1j * BM_c * G
        # exit()
    

    # print('torch.norm(A).item()',torch.norm(A).item())
    # print('A[0:6,0:6]',A[0:6,0:6])
    return A.to(DTYPE)

 
def compute_bs(scatterer: Mesh, board:Tensor, p_ref=Constants.P_ref, norms:Tensor|None=None, a=None, c=None, BM_c=None, k=Constants.k, internal_points=None) -> Tensor:
    '''
    Computes B for the computation of H in the BEM model\n
    :param scatterer: The mesh used (as a `vedo` `mesh` object)
    :param board: Transducers to use 
    :param p_ref: The value to use for p_ref
    :param norms: Tensor of normals for transduers
    :param internal_points: The internal points to use for CHIEF based BEM
    :return B: B tensor
    '''
    centres = torch.tensor(scatterer.cell_centers().points).to(DTYPE).to(device).T.unsqueeze_(0)
    bs = forward_model_batched(centres,board, p_ref=p_ref,norms=norms,k=k) 


    if internal_points is not None:
        F_int = forward_model_batched(internal_points.permute(0,2,1), board, p_ref=p_ref,norms=norms,k=k)
        bs = torch.cat([bs, F_int], dim=1)
    
    if a is not None:
        f_mod = torch.sum(forward_model_batched(a,board, p_ref=p_ref,norms=norms), dim=1, keepdim=True)
        bs += c * f_mod
    
    if BM_c is not None:

        bs = (1-1j * BM_c) * bs
        
        
    return bs   

 
def compute_H(scatterer: Mesh, board:Tensor ,use_LU:bool=True, use_OLS:bool = False, p_ref = Constants.P_ref, norms:Tensor|None=None, k:float=Constants.k, betas:float|Tensor = 0, a=None, c=None, BM_c=None, internal_points=None) -> Tensor:
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

    :return H: H
    '''

    # internal_points = None

    centres = torch.tensor(scatterer.cell_centers().points, dtype=DTYPE, device=device)
    M = centres.shape[0]

    if internal_points is not None and (internal_points.shape[1] == 3 and internal_points.shape[2] != 3):
            internal_points = internal_points.permute(0,2,1)

    A = compute_A(scatterer, betas=betas, a=a, c=c,BM_c=BM_c, k=k,internal_points=internal_points)
    bs = compute_bs(scatterer,board,p_ref=p_ref,norms=norms,a=a,c=c,BM_c=BM_c, k=k,internal_points=internal_points)


    # print(A.shape, bs.shape)
    
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
    return H



def get_cache_or_compute_H(scatterer:Mesh,board,use_cache_H:bool=True, path:str="Media", 
                           print_lines:bool=False, cache_name:str|None=None, p_ref = Constants.P_ref, 
                           norms:Tensor|None=None, method=Literal['OLS','LU', 'INV'], k:float=Constants.k, betas:float|Tensor = 0, a=None, c=None, BM_c=None, internal_points=None) -> Tensor:
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
    :param internal_points: The internal points to use for CHIEF based BEM
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
            cache_name = scatterer.filename+"--"+ board_name(board) + '--' + str(p_ref) + '--' + str(k)
            cache_name = hashlib.md5(cache_name.encode()).hexdigest()
        f_name = path+"/BEMCache/"  +  cache_name + ".bin"
        # print(f_name)

        try:
            if print_lines: print("Trying to load H at", f_name ,"...")
            H = pickle.load(open(f_name,"rb")).to(device).to(DTYPE)
        except FileNotFoundError: 
            if print_lines: print("Not found, computing H...")
            H = compute_H(scatterer,board,use_LU=use_LU,use_OLS=use_OLS,norms=norms, k=k, betas=betas, a=a, c=c,BM_c=BM_c, internal_points=internal_points)
            f = open(f_name,"wb")
            pickle.dump(H,f)
            f.close()
    else:
        if print_lines: print("Computing H...")
        H = compute_H(scatterer,board, p_ref=p_ref,norms=norms,use_LU=use_LU,use_OLS=use_OLS, k=k, betas=betas, a=a, c=c,BM_c=BM_c, internal_points=internal_points)

    return H

def compute_E(scatterer:Mesh, points:Tensor, board:Tensor|None=None, use_cache_H:bool=True, print_lines:bool=False,
               H:Tensor|None=None,path:str="Media", return_components:bool=False, p_ref = Constants.P_ref, norms:Tensor|None=None, H_method=None, 
               k:float=Constants.k, betas:float|Tensor = 0, alphas:float|Tensor=1, a=None, c=None, BM_c=None, internal_points=None) -> Tensor:
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
    :param internal_points: The internal points to use for CHIEF based BEM

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
        H = get_cache_or_compute_H(scatterer,board,use_cache_H, path, print_lines,p_ref=p_ref,norms=norms, method=H_method, k=k, betas=betas, a=a, c=c,BM_c=BM_c, internal_points=internal_points).to(DTYPE)
        
    if print_lines: print("G...")
    G = compute_G(points, scatterer, k=k, betas=betas,alphas=alphas).to(DTYPE)
    
    if print_lines: print("F...")
    F = forward_model_batched(points,board,p_ref=p_ref,norms=norms, k=k).to(DTYPE)  
    # if a is not None:
    #     F += c * forward_model_batched(a,board, p_ref=p_ref,norms=norms)
    
    if print_lines: print("E...")

    E = F+G@H

    torch.cuda.empty_cache()
    if return_components:
        return E.to(DTYPE), F.to(DTYPE), G.to(DTYPE), H.to(DTYPE)
    return E.to(DTYPE)


