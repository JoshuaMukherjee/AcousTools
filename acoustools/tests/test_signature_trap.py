from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_abs, forward_model
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise
from acoustools.Gorkov import gorkov_analytical
from acoustools.Constants import k

import torch

import matplotlib.pyplot as plt

'''
Bk2 Pg 162
'''

p = create_points(1,1,max_pos=0.05, y=0)
board = TRANSDUCERS

x = wgs(p,board=board)

F = forward_model(p, board)


sig = (torch.linspace(1,512,512) * 2*torch.pi/512)
sig = torch.reshape(sig, (2,16,16)).mT
sig = torch.reshape(sig,(512,))



print(torch.angle(torch.sum(torch.exp(1j*sig))))


print(torch.sum(sig[:256]), torch.sum(sig[256:]))
sig = sig.unsqueeze(0).unsqueeze(2)
x = x * torch.e**(1j*sig)
# print(torch.angle(F.mT*x))

A,B,C = ABC(0.06, plane='xz')
Visualise(A,B,C,x,p, colour_functions=[propagate_abs])

A,B,C = ABC(0.06, plane='yz')
# Visualise(A,B,C,x,p, colour_functions=[propagate_abs])
