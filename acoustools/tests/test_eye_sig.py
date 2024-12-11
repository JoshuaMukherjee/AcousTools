from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC

board = TRANSDUCERS

p = create_points(1,1,y=0)
x = wgs(p, board=board)
x = add_lev_sig(x,mode='Eye')

A,B,C = ABC(0.07)

Visualise(A,B,C,x, p)
