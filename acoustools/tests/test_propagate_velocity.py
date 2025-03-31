from acoustools.Utilities import create_points, propagate_velocity_real, propagate_abs, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise


p = create_points(1,1,0,0,0)
x = wgs(p)
x = add_lev_sig(x)


def vel_pot_x(activations, points):
    return propagate_velocity_real(activations, points)[0]

def vel_pot_y(activations, points):
    return propagate_velocity_real(activations, points)[1]

def vel_pot_z(activations, points):
    return propagate_velocity_real(activations, points)[2]


    

Visualise(*ABC(0.1, plane='xz'),x,colour_functions=[propagate_abs,vel_pot_x, vel_pot_y, vel_pot_z], link_ax=[1,2,3])