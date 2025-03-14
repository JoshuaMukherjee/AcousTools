from acoustools.Utilities import create_points, add_lev_sig
from acoustools.Solvers import wgs

p = create_points(1,3)
x = wgs(p)
add_lev_sig(x)