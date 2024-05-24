from acoustools.Utilities import add_lev_sig, create_points, TRANSDUCERS
from acoustools.Visualiser import Visualise_single
from acoustools.Solvers import wgs

import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation



board = TRANSDUCERS
plane = board[:,0:2]

p = create_points(1,1,0,0,0)
x_start = wgs(p)

sig_start = torch.atan2(plane[:,0], plane[:,1]).unsqueeze(0).unsqueeze(2).reshape((1,-1, 256))

radius = (plane[:,0]**2 + plane[:,1]**2)**0.5
angle = torch.atan2(plane[:,0], plane[:,1])

sig = sig_start.clone()


x = add_lev_sig(x_start,sig=sig)


A = torch.tensor((-0.02, 0.02,0))
B = torch.tensor((0.02, 0.02,0))
C = torch.tensor((-0.02, -0.02,0))

D = torch.tensor((-0.02,0, 0.02))
E = torch.tensor((0.02,0, 0.02))
F = torch.tensor((-0.02,0, -0.02))


fig = plt.figure()

ax_vis = fig.add_subplot(2,3,1)
img = Visualise_single(A,B,C,x)
im = ax_vis.matshow(img, cmap='hot')
fig.colorbar(im, ax=ax_vis)
ax_vis.axis('off')

ax_vis_2 = fig.add_subplot(2,3,2)
img_2 = Visualise_single(D,E,F,x)
im_2 = ax_vis_2.matshow(img_2, cmap='hot')
fig.colorbar(im, ax=ax_vis_2)
ax_vis_2.axis('off')



ax_sig_1 = fig.add_subplot(2,3,3)
sig_im_vortex = ax_sig_1.matshow(sig.real.reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax_sig_1.axis('off')


ax_sig_2 = fig.add_subplot(2,3,4)
sig_im_vortex_2 = ax_sig_2.matshow(sig.real.reshape(1,-1,16,16)[0,1,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax_sig_2.axis('off')

ax_holo_1 = fig.add_subplot(2,3,5)
holo_im_1 = ax_holo_1.matshow(torch.angle(x).reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax_holo_1.axis('off')

ax_holo_2 = fig.add_subplot(2,3,6)
holo_im_2 = ax_holo_2.matshow(torch.angle(x).reshape(1,-1,16,16)[0,1,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax_holo_2.axis('off')


def traverse(index):
    # global ax_vis, ax_sig_1, ax_sig_2
    # ax_vis.clear()
    # ax_sig_1.clear()
    # ax_sig_2.clear()

    # mask = radius < 0.01*(index+1)
    # mask = torch.logical_and(radius < 0.01*(index+1), radius > 0.01*(index-1))
    # print(((torch.exp(angle) * torch.cos(angle)) **2 + (torch.exp(angle) * torch.sin(angle))**2)**0.5 - radius)
    # mask = torch.isclose(0.06*torch.max(radius)*torch.exp(angle), torch.remainder(radius,torch.pi*2), 1e-1)
    # mask = torch.remainder(radius,torch.pi*2) < 0.5*torch.max(radius)*((angle+torch.pi)/torch.max(angle + torch.pi))

    R = 0.001*index + 0.005
    mask = torch.logical_and(torch.remainder(radius,R*torch.pi*2) < R*(angle + torch.pi), torch.remainder(radius,(R*2.5)*torch.pi*2) > (R*2.5)*(angle + torch.pi))
    
    # print(mask)

    mask = mask.reshape((1,-1, 256))

    sig = sig_start.clone()


    sig[:,0,:][mask[:,0,:]] = 0
    sig[:,1,:][mask[:,1,:]] = torch.pi

    x = add_lev_sig(x_start,sig=sig)

    im.set_data(Visualise_single(A,B,C,x))
    im_2.set_data(Visualise_single(D,E,F,x))
    sig_im_vortex.set_data(sig.real.reshape(1,-1,16,16)[0,0,:,:].mT)
    sig_im_vortex_2.set_data(sig.real.reshape(1,-1,16,16)[0,1,:,:].mT)
    holo_im_1.set_data(torch.angle(x).reshape(1,-1,16,16)[0,0,:,:].mT)
    holo_im_2.set_data(torch.angle(x).reshape(1,-1,16,16)[0,1,:,:].mT)

    


lap_animation = animation.FuncAnimation(fig, traverse, frames=10, interval=500)

lap_animation.save('Results.gif', dpi=80, writer='imagemagick')


