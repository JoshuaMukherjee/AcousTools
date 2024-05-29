from acoustools.Utilities import create_points, propagate_phase, propagate_abs, device
from acoustools.Solvers import naive
from acoustools.Visualiser import Visualise
import acoustools.Constants as c

import torch

p = create_points(4,1,0,0,0)

DELTA = c.wavelength/2
p[:,0,0] += DELTA
p[:,0,1] -= DELTA
p[:,1,2] += DELTA
p[:,1,3] -= DELTA

print(p)
# 0 3 1 2

activation = torch.ones(p.shape[2],1) + 0j
activation= activation.to(device)
activation[0,:] += torch.e ** (torch.pi/4*1j)
activation[3,:] += torch.e ** (3*torch.pi/4*1j)
activation[1,:] += torch.e ** (5*torch.pi/4*1j)
activation[2,:] += torch.e ** (7*torch.pi/4*1j)

x = naive(p, activation=activation)

A = torch.tensor((-0.02, 0.02,0))
B = torch.tensor((0.02, 0.02,0))
C = torch.tensor((-0.02, -0.02,0))
Visualise(A,B,C,x, colour_functions=[propagate_abs,propagate_phase],cmaps=['hot','hsv'],clr_labels=['Pressure (Pa)','Angle (rad)'])

