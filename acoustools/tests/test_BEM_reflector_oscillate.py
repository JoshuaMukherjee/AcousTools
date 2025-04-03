from acoustools.BEM import load_scatterer, propagate_BEM_pressure, compute_E
from acoustools.Utilities import TOP_BOARD, create_points
from acoustools.Solvers import gorkov_target, wgs
from acoustools.Visualiser import Visualise, ABC
from acoustools.Constants import k

import torch

path = "../BEMMedia"

USE_CACHE = True
board = TOP_BOARD

reflector_path =  path+"/flat-lam2.stl"
reflector = load_scatterer(reflector_path, dz=0.0) #Make mesh at 0,0,0


delta = 0.01
p = create_points(1,1,0,0,0.03)
# p2 = create_points(1,1,delta,0,0.03)
p2 = create_points(1,1,y=0, max_pos=0.04, min_pos=0.01)

# x = gorkov_target(p, board=board, reflector=reflector, path=path)

E,F,G,H = compute_E(reflector,p,return_components=True, path=path)

x = wgs(p, A=E)


distance1 = torch.sqrt(torch.sum((board.unsqueeze(0).mT - p.expand(1,3,256))**2,axis=1,keepdim=True)).permute(0,2,1)
distance2 = torch.sqrt(torch.sum((board.unsqueeze(0).mT - p2.expand(1,3,256))**2,axis=1,keepdim=True)).permute(0,2,1)
distance = distance1 - distance2

x = torch.exp(1j * (torch.angle(x) + distance * k))


Visualise(*ABC(0.04, origin=p), x,points=torch.stack([p, p2],dim=2), colour_functions=[propagate_BEM_pressure], 
          colour_function_args=[{'scatterer':reflector,'board':board,'path':path, 'H':H}], res=(100,100))