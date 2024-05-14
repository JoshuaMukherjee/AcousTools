from acoustools.Utilities import create_points, propagate_abs, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise
from acoustools.Levitator import LevitatorController

import torch

p = create_points(4,1, z=0)
p[:,0,0] = -0.02 
p[:,1,0] = -0.02

p[:,0,1] = 0.02
p[:,1,1] = -0.02 

p[:,0,2] = -0.02
p[:,1,2] = 0.02

p[:,0,3] = 0.02
p[:,1,3] = 0.02


print(p)
x = wgs(p)
print(propagate_abs(x,p))

x = add_lev_sig(x)

A = torch.tensor((-0.09, 0.09,0))
B = torch.tensor((0.09, 0.09,0))
C = torch.tensor((-0.09, -0.09,0))
normal = (0,1,0)
origin = (0,0,0)

# Visualise(A,B,C, x, points=p)

lev = LevitatorController(ids=(73,53))
lev.levitate(x)
print('Levitating...')
input()

print('Stopping...')
lev.disconnect()
print('Stopped')