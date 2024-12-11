from acoustools.Utilities import TRANSDUCERS, create_points, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise_single_blocks, ABC, Visualise
from acoustools.Gorkov import gorkov_analytical

import matplotlib.pyplot as plt
import torch

board = TRANSDUCERS

p = create_points(1,1, min_pos=-0.05, max_pos=0.05)
print(p)

x = wgs(p,board=board)


x,sig = add_lev_sig(x,mode='Eye', return_sig=True)
sig_img = sig.reshape((-1,16,16))

A,B,C = ABC(0.05, origin=p.cpu().detach().squeeze())

res = (200,200)
im= Visualise_single_blocks(A,B,C,x,res=res)
im_U= Visualise_single_blocks(A,B,C,x,res=res, colour_function=gorkov_analytical)


A,B,C = ABC(0.05, plane='xy', origin=p.cpu().detach().squeeze())
res = (200,200)

im_xy= Visualise_single_blocks(A,B,C,x,res=res)
im_xy_U= Visualise_single_blocks(A,B,C,x,res=res, colour_function=gorkov_analytical)


fig, axs = plt.subplots(2,4)

mat = axs[0,0].matshow(im.cpu().detach(),cmap='hot')
plt.colorbar(mat, ax=axs[0,0])

mat_U = axs[0,1].matshow(im_U.cpu().detach(),cmap='hot')
plt.colorbar(mat_U, ax=axs[0,1])


sig1= axs[0,2].matshow(sig_img[0,:,:].cpu().detach(),cmap='hsv')
plt.colorbar(sig1, ax=axs[0,2])


sig2 = axs[0,3].matshow(sig_img[1,:,:].cpu().detach(),cmap='hsv')
plt.colorbar(sig2, ax=axs[0,3])

mat_xy = axs[1,0].matshow(im_xy.cpu().detach(),cmap='hot')
plt.colorbar(mat_xy, ax=axs[1,0])

mat_xy_U = axs[1,1].matshow(im_xy_U.cpu().detach(),cmap='hot')
plt.colorbar(mat_xy_U, ax=axs[1,1])

x_im = torch.angle(x.reshape(-1,16,16))
holo1 = axs[1,2].matshow(x_im[0,:].cpu().detach(),cmap='hsv')
plt.colorbar(holo1, ax=axs[1,2])

holo2 = axs[1,3].matshow(x_im[1,:].cpu().detach(),cmap='hsv')
plt.colorbar(holo2, ax=axs[1,3])


plt.show()



