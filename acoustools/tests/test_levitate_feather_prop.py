from acoustools.Utilities import create_points, propagate_abs, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise
from acoustools.Levitator import LevitatorController

import torch

p = create_points(3,1, y=0)
p[:,:,0] = 0 

p[:,0,1] = 0.015 
p[:,2,1] = 0 

p[:,0,2] = -0.012 
p[:,2,2] = 0.018

print(p)
x = wgs(p)
x = add_lev_sig(x)

A = torch.tensor((-0.09,0, 0.09))
B = torch.tensor((0.09,0, 0.09))
C = torch.tensor((-0.09,0, -0.09))
normal = (0,1,0)
origin = (0,0,0)

Visualise(A,B,C, x, points=p)

lev = LevitatorController(ids=(73,53))
lev.levitate(x)
print('Levitating...')
input()

print('Stopping...')
lev.disconnect()
print('Stopped')