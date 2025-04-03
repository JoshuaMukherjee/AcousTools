from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig
from acoustools.Force import compute_force
from acoustools.Solvers import wgs

board = TRANSDUCERS
p = create_points(1,1)

x = wgs(p, board=board)
x = add_lev_sig(x)

f = compute_force(x,p,board)

print(f)
