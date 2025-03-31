from acoustools.Utilities import create_points, propagate_abs, propagate_speed, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise


p = create_points(1,1,0,0,0)
x = wgs(p)
x = add_lev_sig(x)

Visualise(*ABC(0.1),x,colour_functions=[propagate_abs,propagate_speed], link_ax=[1,2,3])