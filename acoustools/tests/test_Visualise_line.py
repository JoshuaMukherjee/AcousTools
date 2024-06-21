from acoustools.Visualiser import Visualise_line
from acoustools.Utilities import create_points
from acoustools.Solvers import wgs
import torch


p = create_points(1,1,x=0,y=0,z=0)
x = wgs(p)


A = torch.tensor((-0.05,0, 0))
B = torch.tensor((0.05,0, 0))

Visualise_line(A,B, x)