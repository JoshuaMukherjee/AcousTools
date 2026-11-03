import acoustools.Constants as c
from acoustools.Utilities import propagate, forward_model_batched

from acoustools.SphericalHarmonics.SphericalHarmonics import convert_to_spherical, transducer_expansion_coef_Snm, spherical_harmonic_Y, mie_coef
from acoustools.SphericalHarmonics.Bessel import spherical_bessel_j, spherical_bessel_h1, spherical_bessel_j_grad, spherical_bessel_h1_grad

import torch, scipy
from torch import Tensor

from line_profiler import profile

@profile
def spherical_harmonics_propagate(activations: Tensor, points: Tensor,board: Tensor|None=None, centre_of_mass=None, max_n=40, p_ref=c.P_ref, transducer_norms:Tensor=None,k=c.k, transducer_radius=c.radius,
                                  particle_radius = c.R, medium_density=c.p_0, medium_speed=c.c_0, particle_density=c.p_p, particle_speed=c.c_p):
    
    # p_tran = propagate(activations, points, board, p_ref=p_ref, k=k,transducer_radius=transducer_radius, norms=transducer_norms)

    
    p_inc = spherical_harmonics_propagate_inc(activations, points, board, centre_of_mass, max_n, p_ref, transducer_norms, k, transducer_radius)
    p_scat = spherical_harmonics_propagate_scat(activations, points, board, centre_of_mass, max_n, p_ref, transducer_norms, k, transducer_radius, particle_radius, medium_density, medium_speed, particle_density, particle_speed)
    
    # p_inc=0
    # p_scat = 0
    # print(p_inc.shape, p_scat.shape)
    # exit()
    
    pressure = (p_inc + p_scat).squeeze(1)

    
    pressure[pressure.isnan()] = 0
    # pressure[pressure.abs() > upper_limit] = 0
    
    return pressure


@profile
def spherical_harmonics_propagate_inc(activations: Tensor, points: Tensor,board: Tensor|None=None, centre_of_mass=None, max_n=40, p_ref=c.P_ref, transducer_norms:Tensor=None,k=c.k, transducer_radius=c.radius):
    
    distance, theta, phi = convert_to_spherical(points, centre_of_mass)
        
    directivity = forward_model_batched(centre_of_mass, board, p_ref=1, norms = transducer_norms, transducer_radius= transducer_radius, k=k, only_directivity=True)  * p_ref
    board_distance, board_theta, board_phi = convert_to_spherical(board.unsqueeze(0).permute(0,2,1), centre_of_mass)
    # board_distance, board_theta, board_phi = None, None, None
    result = None
    
    for n in range(max_n):
        
        jn = spherical_bessel_j(k * distance, n) 
        
        for m in range(-n,n+1): #n or n + 1?
            # print(n,m)
            
            Ynm = spherical_harmonic_Y(theta, phi, n, m)
            print('inc',board_theta.shape, board_phi.shape)
            Snm = transducer_expansion_coef_Snm(activations, board, centre_of_mass, n, m, transducer_norms=transducer_norms, k=k, transducer_radius=transducer_radius, directivity=directivity, distance=board_distance, theta=board_theta, phi=board_phi) 
            
            # print(scipy.special.sph_harm_y(n,m,theta.real, phi.real).squeeze() / Ynm.squeeze())
   
            product = jn * Snm * Ynm
            
            
            
            if result is None: 
                result = product
            else:
                result += product
            
    

    
    return result


@profile
def spherical_harmonics_propagate_scat(activations: Tensor, points: Tensor,board: Tensor|None=None, centre_of_mass=None, max_n=40, p_ref=c.P_ref, transducer_norms:Tensor=None,k=c.k, 
                                       transducer_radius=c.radius, particle_radius = c.R, medium_density=c.p_0, medium_speed=c.c_0, particle_density=c.p_p, particle_speed=c.c_p):
    
    distance, theta, phi = convert_to_spherical(points, centre_of_mass)
    
    directivity = forward_model_batched(centre_of_mass, board, p_ref=1, norms = transducer_norms, transducer_radius= transducer_radius, k=k, only_directivity=True)  * p_ref
    board_distance, board_theta, board_phi = convert_to_spherical(board.unsqueeze(0).permute(0,2,1), centre_of_mass)
    # board_distance, board_theta, board_phi = None, None, None
    result = None
        
    for n in range(max_n):
        
        hn = spherical_bessel_h1(k * distance, n).unsqueeze(1)
        cn = mie_coef(n, particle_radius, medium_density, medium_speed, particle_density, particle_speed)
        
        for m in range(-n,n+1): #n or n + 1?
            # print(n,m)
            
            
            S_hat = transducer_expansion_coef_Snm(activations, board, centre_of_mass, n, m, transducer_norms=transducer_norms, k=k, transducer_radius=transducer_radius, p_ref=1, directivity=directivity, distance=board_distance, theta=board_theta, phi=board_phi) 
            Snm = cn * S_hat
            
            Ynm = spherical_harmonic_Y(theta, phi, n, m)
            # print(scipy.special.sph_harm_y(n,m,theta.real, phi.real).squeeze() / Ynm.squeeze())
            product = hn * Snm * Ynm
            # exit()
           
           
           
            if result is None: 
                result = product
            else:
                result += product
            
        # print(result)
    
    return result