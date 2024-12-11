from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise_single_blocks, ABC
from acoustools.Gorkov import gorkov_analytical

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


import torch

board = TRANSDUCERS

p = create_points(1,1,0,0,0)

x = wgs(p, board=board)

x_trap = add_lev_sig(x, board, 'Trap')
x_eye = add_lev_sig(x,board,'Eye')

A,B,C = ABC(0.02)

res = (400,400)

im_trap = Visualise_single_blocks(A,B,C, x_trap, res=res).cpu().detach()
im_eye = Visualise_single_blocks(A,B,C, x_eye, res=res).cpu().detach()

im_trap_U = Visualise_single_blocks(A,B,C, x_trap, res=res, colour_function=gorkov_analytical).cpu().detach()
im_eye_U = Visualise_single_blocks(A,B,C, x_eye, res=res, colour_function=gorkov_analytical).cpu().detach()

vmax = torch.max(torch.concat([im_trap,im_eye]))
vmin = torch.min(torch.concat([im_trap,im_eye]))
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

vmaxU = torch.max(torch.concat([im_trap_U,im_eye_U]))
vminU = torch.min(torch.concat([im_trap_U,im_eye_U]))
normU = mcolors.Normalize(vmin=vminU, vmax=vmaxU)

fig, axs = plt.subplots(2,2)

a = axs[0,0].matshow(im_trap, cmap='hot', norm=norm)
divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(a,cax=cax)

b =axs[0,1].matshow(im_eye, cmap='hot', norm=norm)
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(b, cax=cax)

c = axs[1,0].matshow(im_trap_U, cmap='hot', norm=normU)
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(c,cax=cax)

d =axs[1,1].matshow(im_eye_U, cmap='hot', norm=normU)
divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(d, cax=cax)

plt.show()