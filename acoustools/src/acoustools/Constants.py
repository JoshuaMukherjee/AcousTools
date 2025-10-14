
pi = 3.1415926535
'''pi to 10 decimal places'''
R = .001 
'''Radius of particle'''
V = 4/3 * pi * R**3
'''Volume of Particle'''
c_0 = 343
'''Speed of sound in air'''
p_0 = 1.2
'''Density of air'''
c_p = 1052
'''Speed of sound in EPS particle'''
p_p = 29.36 
'''density of EPS particle, From `Holographic acoustic elements for manipulation of levitated objects`'''
f = 40000
'''Frequency of 40KHz sound'''
wavelength = c_0 / f #0.008575
'''Wavelength of 40KHz sound'''
k = 2*pi / wavelength #732.7329804081634
'''Wavenumber of 40KHz sound'''
k_eps= k + 0.01j*k
'''Damped Wavenumber of 40KHz sound'''
# radius=0.005 
radius = 0.0045
'''Radius of transducer'''
# P_ref = 8.02 #old value
P_ref = 0.17*20 #3.4
# P_ref = 5.9983 #Value from Zak
'''Reference pressure for transducers'''
angular_frequency = f * 2 * pi 
'''Angular Frequency of 40kHz sound'''