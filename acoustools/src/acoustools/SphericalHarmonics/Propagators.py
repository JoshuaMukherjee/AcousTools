import acoustools.Constants as c

from acoustools.SphericalHarmonics.SphericalHarmonics import convert_to_spherical, transducer_expansion_coef_Snm, spherical_harmonic_Y, mie_coef
from acoustools.SphericalHarmonics.Bessel import spherical_bessel_j, spherical_bessel_h1

import torch, scipy
from torch import Tensor

def spherical_harmonics_propagate(activations: Tensor, points: Tensor,board: Tensor|None=None, centre_of_mass=None, max_n=40, p_ref=c.P_ref, transducer_norms:Tensor=None,k=c.k, transducer_radius=c.radius,
                                  particle_radius = c.R, medium_density=c.p_0, medium_speed=c.c_0, particle_density=c.p_p, particle_speed=c.c_p):
    
    p_inc = spherical_harmonics_propagate_inc(activations, points, board, centre_of_mass, max_n, p_ref, transducer_norms, k, transducer_radius)
    p_scat = spherical_harmonics_propagate_scat(activations, points, board, centre_of_mass, max_n, p_ref, transducer_norms, k, transducer_radius, particle_radius, medium_density, medium_speed, particle_density, particle_speed)
    
    print(p_inc.abs() ,  p_scat.abs())
    print(p_inc ,  p_scat)
    
    return p_inc + p_scat



def spherical_harmonics_propagate_inc(activations: Tensor, points: Tensor,board: Tensor|None=None, centre_of_mass=None, max_n=40, p_ref=c.P_ref, transducer_norms:Tensor=None,k=c.k, transducer_radius=c.radius):
    
    distance, theta, phi = convert_to_spherical(points, centre_of_mass)
    
    result = None
    
    for n in range(max_n):
        for m in range(-n,n+1): #n or n + 1?
            # print(n,m)
            jn = spherical_bessel_j(k * distance, n) 
            Snm = transducer_expansion_coef_Snm(activations, board, centre_of_mass, n, m, transducer_norms=transducer_norms, k=k, transducer_radius=transducer_radius) 
            Ynm = spherical_harmonic_Y(theta, phi, n, m)
            # print(scipy.special.sph_harm_y(n,m,theta.real, phi.real).squeeze() / Ynm.squeeze())
            product = jn * Snm * Ynm
            
            if result is None: 
                result = product
            else:
                result += product
            
        # print(result)
    
    return result



def spherical_harmonics_propagate_scat(activations: Tensor, points: Tensor,board: Tensor|None=None, centre_of_mass=None, max_n=40, p_ref=c.P_ref, transducer_norms:Tensor=None,k=c.k, 
                                       transducer_radius=c.radius, particle_radius = c.R, medium_density=c.p_0, medium_speed=c.c_0, particle_density=c.p_p, particle_speed=c.c_p):
    
    distance, theta, phi = convert_to_spherical(points, centre_of_mass)
    
    result = None
    
    for n in range(max_n):
        for m in range(-n,n+1): #n or n + 1?
            # print(n,m)
            hn = spherical_bessel_h1(k * distance, n) 
            cn = mie_coef(n, particle_radius, medium_density, medium_speed, particle_density, particle_speed)
            Snm = cn * transducer_expansion_coef_Snm(activations, board, centre_of_mass, n, m, transducer_norms=transducer_norms, k=k, transducer_radius=transducer_radius) 
            Ynm = spherical_harmonic_Y(theta, phi, n, m)
            # print(scipy.special.sph_harm_y(n,m,theta.real, phi.real).squeeze() / Ynm.squeeze())
            product = hn * Snm * Ynm
            
           
           
           
            if result is None: 
                result = product
            else:
                result += product
            
        # print(result)
    
    return result