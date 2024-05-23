from acoustools.Utilities import create_points, add_lev_sig, propagate_abs, TRANSDUCERS, BOTTOM_BOARD
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise_single

import matplotlib.pyplot as plt

import torch


board = TRANSDUCERS

p = create_points(1,x=0,y=0,z=0)
x = wgs(p, board=board)


modes = ['Trap', 'Focal', 'Vortex', 'Twin']
N = len(modes)

fig = plt.figure()

A = torch.tensor((-0.02, 0.02,0))
B = torch.tensor((0.02, 0.02,0))
C = torch.tensor((-0.02, -0.02,0))

D = torch.tensor((-0.02,0, 0.02))
E = torch.tensor((0.02,0, 0.02))
F = torch.tensor((-0.02,0, -0.02))


M=6
for i,mode in enumerate(modes):

    x_sig, sig = add_lev_sig(x.clone(), mode=mode, return_sig=True, board=board)
    

    ax= fig.add_subplot(N,M,M*(i)+1)
    if i == 0: ax.set_title("Pressure XY")
    img = Visualise_single(A,B,C, x_sig, colour_function_args={"board":board})
    img_p = ax.matshow(img, cmap='hot', vmax=9000,vmin=0)
    fig.colorbar(img_p, ax=ax)
    ax.axis('off')

    ax= fig.add_subplot(N,M,M*i+2)
    if i == 0: ax.set_title("Pressure XZ")
    img = Visualise_single(D,E,F, x_sig, colour_function_args={"board":board})
    img_p = ax.matshow(img, cmap='hot', vmax=9000,vmin=0)
    fig.colorbar(img_p, ax=ax)
    ax.axis('off')

    ax= fig.add_subplot(N,M,M*i+3)
    if i == 0:  ax.set_title("Signature Top")
    sig_im = ax.matshow(sig.real.reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sig_im, ax=ax)
    ax.axis('off')

    ax= fig.add_subplot(N,M,M*i+4)
    if i == 0:  ax.set_title("Signature Bottom")
    sig_im = ax.matshow(sig.real.reshape(1,-1,16,16)[0,1,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sig_im, ax=ax)
    ax.axis('off')

    ax = fig.add_subplot(N,M,M*i+5)
    if i == 0: ax.set_title("Hologram Top")
    holo_im= ax.matshow(torch.angle(x_sig).reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(holo_im, ax=ax)
    ax.axis('off')

    ax = fig.add_subplot(N,M,M*i+6)
    if i == 0: ax.set_title("Hologram Bottom")
    holo_im= ax.matshow(torch.angle(x_sig).reshape(1,-1,16,16)[0,1,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(holo_im, ax=ax)
    ax.axis('off')


plt.show()