from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise
from acoustools.Gorkov import gorkov_analytical

import torch

p = create_points(1,1,0,0,0)
board = TRANSDUCERS

x = wgs(p,board=board)

act = torch.reshape(x,(1,-1, 256))
sig = torch.cat([torch.ones_like(act[:,0,:]) * torch.pi, torch.zeros_like(act[:,0,:])])
print(sig.shape)
perm = torch.randperm(512)

sig = sig.reshape(512)[perm]

sig = sig.reshape(2,256)


x = add_lev_sig(x, board, sig=sig)

A,B,C = ABC(0.06)
Visualise(A,B,C,x,p, colour_functions=[propagate_abs, gorkov_analytical])