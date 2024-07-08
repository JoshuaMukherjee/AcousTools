from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD, create_board, propagate, device, propagate_phase
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength
from acoustools.Visualiser import Visualise_single
from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_H, propagate_BEM_pressure

import torch

import matplotlib.animation as animation
import matplotlib.pyplot as plt


z = 2*wavelength

path = '../../BEMMedia/'

reflector = load_scatterer('flat-lam2.stl',root_path=path, dz=-z)

board = create_board(2,z=z)

x = torch.ones((1,1)) * torch.e**(1j*0)
# x[1,:] = torch.ones((1,1)) * torch.e**(1j*torch.pi)


dz = 0.001
A = torch.tensor((-z,0, z-dz))
B = torch.tensor((z,0, z-dz))
C = torch.tensor((-z,0, -z+dz))


fig = plt.figure()
ax1 = fig.add_subplot(111)


FRAMES = 10

def traverse(index):

    board = create_board(2,z=z - (wavelength/FRAMES) * index)

    print(board)

    im = Visualise_single(A,B,C, x, colour_function=propagate_BEM_pressure, colour_function_args={'board':board,'scatterer':reflector,'path':path}, res=(200,200))

    ax1.matshow(im, cmap = 'hot', vmax=300)


animation = animation.FuncAnimation(fig, traverse, frames=FRAMES, interval=500)
# plt.show()

animation.save('Results.gif', dpi=80, writer='imagemagick')