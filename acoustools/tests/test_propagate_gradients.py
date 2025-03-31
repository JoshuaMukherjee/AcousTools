from acoustools.Utilities import create_points, propagate_pressure_grad, propagate_abs, add_lev_sig, TRANSDUCERS
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise


p = create_points(1,1,0,0,0)
x = wgs(p)
x = add_lev_sig(x)


def vel_grad_x(activations, points):
    return propagate_pressure_grad(activations, points,TRANSDUCERS)[0].real

def vel_grad_y(activations, points):
    return propagate_pressure_grad(activations, points,TRANSDUCERS)[1].real

def vel_grad_z(activations, points):
    return propagate_pressure_grad(activations, points,TRANSDUCERS)[2].real


    

Visualise(*ABC(0.1),x,colour_functions=[propagate_abs,vel_grad_x, vel_grad_y, vel_grad_z], link_ax=[1,2,3])