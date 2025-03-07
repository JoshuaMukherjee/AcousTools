from acoustools.Utilities import TOP_BOARD, device, DTYPE, create_points
from acoustools.Mesh import load_scatterer
from acoustools.Solvers import gorkov_target
from acoustools.BEM import propagate_BEM_pressure
from acoustools.Visualiser import Visualise, ABC

import torch

root = "../BEMMedia/" #Change to path to BEMMedia Folder
path = root+"flat-lam2.stl"

reflector = load_scatterer(path) #Change dz to be the position of the reflector


board = TOP_BOARD
U_target = torch.tensor([-7.5e-6,]).to(device).to(DTYPE)

p = create_points(1,1,0,0,0.03)

x = gorkov_target(p,board=board, U_targets=U_target, reflector=reflector, path=root)

abc = ABC(0.05)
Visualise(*abc, x, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':reflector,'path':root}])