from acoustools.BEM import load_scatterer, scatterer_file_name, compute_E, propagate_BEM_pressure, BEM_forward_model_grad, BEM_gorkov_analytical
from acoustools.Mesh import get_lines_from_plane
from acoustools.Utilities import create_points, TRANSDUCERS, device, add_lev_sig, forward_model_grad, propagate_abs, TOP_BOARD, BOTTOM_BOARD
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise, ABC
import acoustools.Constants as Constants
from acoustools.Gorkov import gorkov_analytical

import vedo, torch
path = "../BEMMedia"

USE_CACHE = True
board = BOTTOM_BOARD

sphere_pth =  path+"/Sphere-lam2.stl"
sphere = load_scatterer(sphere_pth, dy=-0.06) #Make mesh at 0,0,0

# vedo.show(sphere, axes=1)
# exit()

N = 1
B = 1

# p = create_points(N,B,y=0)
# p = create_points(N,B,y=0,x=0,z=-0.04)
p = create_points(N,B, z=-0.04, y=0, x=0)
d_lambda = create_points(N,B,0,0,Constants.wavelength/4)
print(p,p-d_lambda)
# p = torch.tensor([[0,0],[0,0],[-0.06]]).unsqueeze(0).to(device)


E,F,G,H = compute_E(sphere, p, board=board, path=path, use_cache_H=USE_CACHE, return_components=True)
x = wgs(p, A=E)

print(torch.abs(E@x))

U = BEM_gorkov_analytical(x, p-d_lambda, sphere, board, path=path)
print(U)
U_pm = gorkov_analytical(x, p-d_lambda, board)
print(U_pm)

abc = ABC(0.06)
Visualise(*abc, x,points=p, colour_functions=[propagate_BEM_pressure,], colour_function_args=[{'board':board,'scatterer':sphere, 'path':path}], res=(200,200))