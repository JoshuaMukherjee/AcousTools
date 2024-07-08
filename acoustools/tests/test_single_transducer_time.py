from acoustools.Utilities import green_propagator, create_points, TRANSDUCERS, propagate_abs, add_lev_sig, BOTTOM_BOARD, create_board, propagate, device, propagate_phase
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength, angular_frequency
from acoustools.Visualiser import Visualise_single

import torch

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



z = wavelength
t1 = create_board(2,z=-z)

board = t1

print(board)

x = torch.ones((1,1)) * torch.e**(1j*0)
# x[1,:] = torch.ones((1,1)) * torch.e**(1j*torch.pi)


dz = 0.001
A = torch.tensor((-z,0, z-dz))
B = torch.tensor((z,0, z-dz))
C = torch.tensor((-z,0, -z+dz))



def propagate_abs_time(activations, points,board=TRANSDUCERS, A=None, A_function=None, A_function_args={}, time=0):
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
    return torch.abs(torch.abs(out) * (torch.e**(-1j * angular_frequency * time)).real)


# Visualise(A,B,C, x, points=p,colour_functions=[propagatae_abs,propagate_abs], colour_function_args=[{"A_function":green_propagator},{}], res=(200,200))

fig = plt.figure()
ax = fig.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)


FRAMES  = 10
def traverse(index):

    print(board)
    time = (angular_frequency/FRAMES) * index


    im = Visualise_single(A,B,C, x, colour_function=propagate_abs_time, colour_function_args={'board':board,'time':time}, res=(100,100))
    image =ax.matshow(im, cmap='hot', vmax=500, vmin = 0)
    fig.colorbar(image, cax=cax, orientation='vertical')



animation = animation.FuncAnimation(fig, traverse, frames=FRAMES, interval=100)
# plt.show()

animation.save('Results.gif', dpi=80, writer='imagemagick')