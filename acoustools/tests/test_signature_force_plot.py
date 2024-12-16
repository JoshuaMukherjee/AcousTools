from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC

from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force

p = create_points(1,1,0,0,0)

board = TRANSDUCERS

x = wgs(p,board=board)
x = add_lev_sig(x, mode = 'Eye')

A,B,C = ABC(0.02, plane='xy')

def force_x(activations, points, board=board, return_components = True):
    return compute_force(activations, points, board, return_components)[0]

def force_y(activations, points, board=board, return_components = True):
    return compute_force(activations, points, board, return_components)[1]

def force_z(activations, points, board=board, return_components = True):
    return compute_force(activations, points, board, return_components)[2]


Visualise(A,B,C,x, colour_functions=[propagate_abs, gorkov_analytical, force_x, force_y], link_ax=None, clr_labels=['Pressure (Pa)','Gorkov','Force (N)', 'Force (N)'])