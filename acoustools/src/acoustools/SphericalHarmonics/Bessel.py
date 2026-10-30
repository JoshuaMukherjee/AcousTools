import scipy, torch, math
import numpy as np
import acoustools.Constants as c

def bessel_J(x, alpha, M=20):
    # total = 0
    # for m in range(M):
    #     total += (((-1) ** m) / (math.factorial(m) * scipy.special.gamma(m + alpha + 1))) * ((x/2) ** (2*m + alpha))
    
    # print(total, scipy.special.jv(alpha, x))
    # print()
    return scipy.special.jv(alpha, x)
    
def bessel_Y(x, alpha, M=20):
    Y = (bessel_J(x, alpha, M) * np.cos(alpha * c.pi)  - bessel_J(x, -1 * alpha, M)) / np.sin(alpha * c.pi)
    return Y

def bessel_H1(x, alpha, M=20):
    H1 = bessel_J(x,alpha,M) + 1j * bessel_Y(x, alpha, M)
    return H1

def spherical_bessel_j(x, alpha, M=20):
    j = np.sqrt(c.pi / (2 * x)) * bessel_J(x, alpha+1/2, M)
    # print('j', j, scipy.special.spherical_jn(alpha, x))
    return j

def spherical_bessel_j_grad(x, alpha, M=20):
    jgrad =  spherical_bessel_j(x, alpha -1, M=M) - (alpha+1)/2 * spherical_bessel_j(x, alpha, M=M) 
    return jgrad

def spherical_bessel_y(x, alpha, M=20):
    y = np.sqrt(c.pi / (2 * x)) * bessel_Y(x, alpha+1/2, M)    
    # print('y', y, scipy.special.spherical_yn(alpha, x))
    return y

def spherical_bessel_y_grad(x, alpha, M=20):
    ygrad =  spherical_bessel_y(x, alpha -1, M=M) - (alpha+1)/2 * spherical_bessel_y(x, alpha, M=M) 
    return ygrad

def spherical_bessel_h1(x, alpha, M=20):
    h1 = spherical_bessel_j(x, alpha, M) + 1j * spherical_bessel_y(x, alpha, M)
    return h1
    
def spherical_bessel_h1_grad(x, alpha, M=20):
    h1grad = spherical_bessel_h1(x, alpha -1, M=M) - (alpha+1)/2 * spherical_bessel_h1(x, alpha, M=M) 
    return h1grad
    
