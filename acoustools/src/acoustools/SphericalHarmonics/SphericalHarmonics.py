
import scipy, torch, math
import acoustools.Constants as c
from acoustools.Utilities import DTYPE, device, forward_model_batched

from acoustools.SphericalHarmonics.Bessel import spherical_bessel_h1,spherical_bessel_h1_grad, spherical_bessel_j, spherical_bessel_j_grad

from line_profiler import profile

_CACHE_Pmn = {}
hashes = {}

def convert_to_spherical(points, origin):
    vecs = origin-points
    
    distance = torch.norm(vecs, dim=1, p=2)
    
    vecs_x = vecs[:,0]
    vecs_y = vecs[:,1]
    vecs_z = vecs[:,2]

    theta = torch.acos(vecs_z / distance).to(DTYPE)
    phi = torch.atan2(vecs_y.real, vecs_x.real).to(DTYPE)
    
    theta[theta.isnan()] = -1*c.pi/2
    
    # print(theta.real.min(), theta.real.max())
    # print(phi.real.min(), phi.real.max())
    # print()
    
    
    # xy_distance = torch.sqrt(vecs_y**2 + vecs_x**2)
    # phi = torch.sgn(vecs_y.real) * torch.arccos(vecs_x / xy_distance)
    
    # print(distance, theta, phi)
    print('Spherical',phi.shape, theta.shape)
        
    return distance, theta, phi
    
    
@profile
def associated_legendre_polynomial(z,n,m):
    '''
    For now uses SciPy - Later come back and write this in torch so Autograd works. 
    '''
    
    print('zp',z.shape)
    hash_z = hash(str(torch.hash_tensor(z.real).item()) + str(z.shape))
    # print(hash_z)
    
    if (n,m,hash_z) in _CACHE_Pmn: 
        P = _CACHE_Pmn[(n,m,hash_z)]
        print('ZP',P.shape,z.shape)
        return P
    else:
        # print(( (n-1,m,hash_z) in _CACHE_Pmn), ((n-2,m,hash_z) in _CACHE_Pmn))
        if( (n-1,m,hash_z) in _CACHE_Pmn) and ((n-2,m,hash_z) in _CACHE_Pmn):
            Pn1 = _CACHE_Pmn[(n-1,m,hash_z)]
            Pn2 = _CACHE_Pmn[(n-2,m,hash_z)]
            
            Pn1 = z*(2*n - 1) * Pn1
            Pn2 = (n + m - 1) * Pn2
            
            P = (Pn1 - Pn2) / (n-m)
        else:
            P = scipy.special.assoc_legendre_p(n,m,z)
            P = torch.tensor(P, device=device, dtype=DTYPE)
    
    _CACHE_Pmn[(n,m,hash_z)] = P    
    print('ZP',P.shape,z.shape)
    return P

@profile
def spherical_harmonic_Y(theta, phi,n,m):
    
    cos_theta = torch.cos(theta)
    P_n_m = associated_legendre_polynomial(cos_theta, n, m)
    print('Y',phi.shape, theta.shape, cos_theta.shape, P_n_m.shape)
    exp = torch.exp(1j * m * phi)
    # exit()
    
    factorial_minus = math.factorial(n - m)
    factorial_plus = math.factorial(n + m)
    
    constant = math.sqrt(((2*n+1) / (4*c.pi)) * (factorial_minus / factorial_plus) )
    
    return constant * P_n_m * exp

@profile
def transducer_expansion_coef_Snm(activations, board, centre_of_mass, n, m, transducer_norms = None, k=c.k, transducer_radius=c.radius, sum_j=True, p_ref = c.P_ref, directivity=None, distance=None, theta=None, phi=None):
    
    if directivity is None:
        directivity = forward_model_batched(centre_of_mass, board, p_ref=1, norms = transducer_norms, transducer_radius= transducer_radius, k=k, only_directivity=True)  * p_ref
    
    if distance is None and theta is None and phi is None:
        print('11')
        distance, theta, phi = convert_to_spherical(board.unsqueeze(0).permute(0,2,1), centre_of_mass)
    
    print('Snm',theta.shape, phi.shape)
    Y = spherical_harmonic_Y(theta, phi, n, m).permute(0,2,1)
    # print(scipy.special.sph_harm_y(n,m,theta.real, phi.real).squeeze() / Y.squeeze())
    # print(Y.shape)
    
    h = spherical_bessel_h1(k * distance, n).unsqueeze(2)

    # hnm = lambda x, n: scipy.special.spherical_jn(n,x) + 1j * scipy.special.spherical_yn(n,x)
    # print(h.squeeze() / hnm(k*distance, n).squeeze())
        
    S_jnm = activations * 4 * c.pi * 1j * k * h * Y.conj() * directivity

    
    if sum_j: return torch.sum(S_jnm, dim=1)
    return S_jnm
    
def mie_coef(n,particle_radius, medium_density=c.p_0, medium_speed=c.c_0, particle_density=c.p_p, particle_speed=c.c_p, k=c.k):

    ka= k * particle_radius
    ka_p = (medium_speed / particle_speed) * ka
    
    jn = spherical_bessel_j(ka, n)
    jn_grad = spherical_bessel_j_grad(ka, n)
    
    hn = spherical_bessel_h1(ka, n)
    # print((scipy.special.spherical_jn(n, ka) + 1j * scipy.special.spherical_yn(n, ka)) / hn)
    hn_grad = spherical_bessel_h1_grad(ka, n)
    
    jn_p = spherical_bessel_j(ka_p, n)
    jn_grad_p = spherical_bessel_j_grad(ka_p, n)
    
    Z = (particle_density * particle_speed) / (medium_density * medium_speed)
    # Z = 1
    
    
    cn =  ((jn*jn_grad_p) - Z * (jn_grad * jn_p)) / (hn*jn_grad_p - Z * (hn_grad * jn_p))
    # 
    cn = cn * -1 
    
    return cn
    
    