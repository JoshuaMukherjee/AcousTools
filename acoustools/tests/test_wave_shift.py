from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_phase, forward_model
from acoustools.Solvers import wgs
from acoustools.Constants import k
from acoustools.Visualiser import ABC, Visualise

import torch, random

'''
Demo of proof in Bk2. Pg 160
'''

board = TRANSDUCERS

p = create_points(1,1,0,0,0)
x = wgs(p, board=board)


p1 = p.clone()
p = p.reshape(1,3).expand((512,3))
distances = torch.sqrt(torch.sum((board - p)**2, axis=1))


p2 = create_points(1,1,max_pos=0.05, y=0)
point = p2.clone()
p2 = p2.reshape(1,3).expand((512,3))
distances2 = torch.sqrt(torch.sum((board - p2)**2, axis=1))

change_dist = distances2 - distances

phases = -1*change_dist * k

phases = phases.unsqueeze_(0).unsqueeze_(2)
x = x * torch.e**(1j*phases)



A,B,C = ABC(0.06)
Visualise(A,B,C,x, torch.stack([p1, point],axis=2))