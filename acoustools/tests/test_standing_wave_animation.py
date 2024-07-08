from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD, create_board, propagate, device, propagate_phase
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength
from acoustools.Visualiser import Visualise_single

import torch

import matplotlib.animation as animation
import matplotlib.pyplot as plt


z = 3*wavelength
t1 = create_board(2,z=-z)
t2 = create_board(2,z=z)

board = torch.cat((t1,t2),axis=0).to(device)

print(board)

x = torch.ones((2,1)) * torch.e**(1j*0)
# x[1,:] = torch.ones((1,1)) * torch.e**(1j*torch.pi)


dz = 0.001
A = torch.tensor((-z,0, z-dz))
B = torch.tensor((z,0, z-dz))
C = torch.tensor((-z,0, -z+dz))




def propagate_abs_norm(activations, points,board=TRANSDUCERS, A=None, A_function=None, A_function_args={}):
    '''
    Propagates a hologram to target points and returns pressure - Same as `torch.abs(propagate(activations, points,board, A))`\\
    `activations` Hologram to use\\
    `points` Points to propagate to\\
    `board` The Transducer array, default two 16x16 arrays\\
    `A` The forward model to use, if None it is computed using `forward_model_batched`. Default:`None`\\
    Returns point pressure
    '''
    if A_function is not None:
        A = A_function(points, board, **A_function_args)

    out = propagate(activations, points,board,A=A)
    return torch.abs(out) / torch.max(torch.abs(out))


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Visualise(A,B,C, x, points=p,colour_functions=[propagatae_abs,propagate_abs], colour_function_args=[{"A_function":green_propagator},{}], res=(200,200))

FRAMES = 10

def traverse(index):

    t1 = create_board(2,z=-z + (wavelength/FRAMES) * index)
    t2 = create_board(2,z=z - (wavelength/FRAMES) * index)

    board = torch.cat((t1,t2),axis=0).to(device)

    print(board)

    x = torch.ones((2,1)) * torch.e**(1j*0)

    im = Visualise_single(A,B,C, x, colour_function=propagate_abs, colour_function_args={'board':board}, res=(200,200))
    im_phase = Visualise_single(A,B,C, x, colour_function=propagate_phase, colour_function_args={'board':board}, res=(200,200))

    ax1.matshow(im, cmap = 'hot', vmax=1000)
    ax2.matshow(im_phase, cmap = 'hsv')


animation = animation.FuncAnimation(fig, traverse, frames=FRAMES, interval=500)
# plt.show()

animation.save('Results.gif', dpi=80, writer='imagemagick')