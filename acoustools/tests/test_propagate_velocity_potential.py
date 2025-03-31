from acoustools.Utilities import create_points, propagate_velocity_potential, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise


p = create_points(1,1,0,0,0)
x = wgs(p)

def vel_pot_real(activations, points):
    return propagate_velocity_potential(activations, points).real

def vel_pot_imag(activations, points):
    return propagate_velocity_potential(activations, points).imag
    

Visualise(*ABC(0.1),x,colour_functions=[propagate_abs,vel_pot_real, vel_pot_imag], link_ax=[1,2])