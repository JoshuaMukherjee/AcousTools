import scipy, torch, math
import numpy as np
import acoustools.Constants as c

from line_profiler import profile


_CACHE_J = {}

@profile
def bessel_J(x, alpha, M=20):
    
    if type(x) == torch.Tensor:
        x_key = torch.hash_tensor(x).item()
    else:
        x_key = x
    
    if (x_key, alpha) in _CACHE_J:
        return _CACHE_J[(x_key, alpha)]
    
    if alpha >= -1 and alpha <= 1:
        J =  scipy.special.jv(alpha, x)
        _CACHE_J[(x_key, alpha)] = J
        return J
    
    if alpha < 0:

        Jplus = bessel_J(x, alpha+1, M=M)
        _CACHE_J[(x_key, alpha+1)] = Jplus
        
        Jplus_2 = bessel_J(x, alpha+2, M=M)
        _CACHE_J[(x_key, alpha+2)] = Jplus_2
        
        J = (2*(alpha+1))/x * Jplus - Jplus_2
        _CACHE_J[(x_key, alpha)] = J
    
    else:      
        Jminus = bessel_J(x, alpha-1, M=M)
        _CACHE_J[(x_key, alpha-1)] = Jminus
        
        Jminus_2 = bessel_J(x, alpha-2, M=M)
        _CACHE_J[(x_key, alpha-2)] = Jminus_2
        
        J = (2*(alpha-1))/x * Jminus - Jminus_2
        _CACHE_J[(x_key, alpha)] = J

    
    # print( J / scipy.special.jv(alpha, x) )
    # exit()
    return J
    
    
def bessel_Y(x, alpha, M=20):
    Y = (bessel_J(x, alpha, M) * np.cos(alpha * c.pi)  - bessel_J(x, -1 * alpha, M)) / np.sin(alpha * c.pi)
    return Y

def bessel_H1(x, alpha, M=20):
    H1 = bessel_J(x,alpha,M) + 1j * bessel_Y(x, alpha, M)
    return H1
@profile
def spherical_bessel_j(x, alpha, M=20, spherical_x = None):
    if spherical_x is None: spherical_x = np.sqrt(c.pi / (2 * x)) 
    j = spherical_x * bessel_J(x, alpha+1/2, M)
    # print('j', j, scipy.special.spherical_jn(alpha, x))
    return j

def spherical_bessel_j_grad(x, alpha, M=20):    
    # jgrad =  spherical_bessel_j(x, alpha -1, M=M) - (alpha+1)/x * spherical_bessel_j(x, alpha, M=M) 
    jgrad =   -1 * spherical_bessel_j(x, alpha+1, M) + (alpha / x) * spherical_bessel_j(x, alpha, M)
    # print(scipy.special.spherical_jn(alpha, x, True) / jgrad)
    
    return jgrad
@profile
def spherical_bessel_y(x, alpha, M=20, spherical_x = None):
    # if alpha <0: print('Y AAAH')
    if spherical_x is None:  spherical_x = np.sqrt(c.pi / (2 * x)) 
    y = spherical_x * bessel_Y(x, alpha+1/2, M)    
    # print('y', y, scipy.special.spherical_yn(alpha, x))
    return y

def spherical_bessel_y_grad(x, alpha, M=20):
    # ygrad =  spherical_bessel_y(x, alpha -1, M=M) - (alpha+1)/x * spherical_bessel_y(x, alpha, M=M) 
    ygrad = -1 * spherical_bessel_y(x, alpha+1, M) + (alpha / x) * spherical_bessel_y(x, alpha, M)

    return ygrad

@profile
def spherical_bessel_h1(x, alpha, M=20):
    spherical_x = np.sqrt(c.pi / (2 * x)) 
    h1 = spherical_bessel_j(x, alpha, M, spherical_x=spherical_x) + 1j * spherical_bessel_y(x, alpha, M, spherical_x=spherical_x)
    return h1
    
def spherical_bessel_h1_grad(x, alpha, M=20):
    
    
    # h1grad_ = spherical_bessel_h1(x, alpha -1, M=M) - (alpha+1)/x * spherical_bessel_h1(x, alpha, M=M) 
    h1grad = -1 * spherical_bessel_h1(x, alpha+1, M) + (alpha / x) * spherical_bessel_h1(x, alpha, M)
    
    return h1grad
    
