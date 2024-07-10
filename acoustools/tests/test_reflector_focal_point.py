from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD, create_board, propagate, device, propagate_phase, TOP_BOARD
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength
from acoustools.Visualiser import Visualise
from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, propagate_BEM_pressure

import torch


z = 2*wavelength

path = '../BEMMedia/'

reflector = load_scatterer('flat-lam2.stl',root_path=path, dz=-z)


board = TOP_BOARD

p = create_points(1,1, x=0,y=0,z=0)

print(p)

E = compute_E(reflector, p, board=board, path=path )


x = wgs(p, board=board,A=E )
x = add_lev_sig(x, board, mode='Twin')



A = torch.tensor((-0.06,0, 0.06))
B = torch.tensor((0.06,0, 0.06))
C = torch.tensor((-0.06,0, -0.06))


im = Visualise(A,B,C, x,p, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'board':board,'scatterer':reflector,'path':path}], res=(200,200))
