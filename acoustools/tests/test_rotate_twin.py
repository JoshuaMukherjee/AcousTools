from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import ABC, Visualise_single_blocks

import matplotlib.pyplot as plt
import torch
import torchvision

board = TRANSDUCERS

p = create_points(1,1,0,0,0)

x = wgs(p, board=board)

x, sig = add_lev_sig(x, board, mode = 'Twin', return_sig=True)


A,B,C = ABC(0.02, plane='xy')

img = Visualise_single_blocks(A,B,C,x)

sig_img = sig.reshape(2,16,16).real.cpu().detach()
rot = torchvision.transforms.RandomRotation((45,45))
sig_img = rot(sig_img)

plt.subplot(1,3,1)

plt.matshow(sig_img[0,:], cmap='hsv', fignum=0, vmax=torch.pi, vmin=-1*torch.pi)

plt.subplot(1,3,2)

plt.matshow(sig_img[1,:], cmap='hsv', fignum=0, vmax=torch.pi, vmin=-1*torch.pi)

plt.subplot(1,3,3)

plt.matshow(img, fignum=0, cmap='hot')

plt.show()