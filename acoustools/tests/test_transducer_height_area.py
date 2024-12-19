from acoustools.Utilities import create_points, BOARD_POSITIONS, transducers, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Force import compute_force

import matplotlib.pyplot as plt


import torch


min_x = -0.05
max_x = 0.05
x_step = 0.01
Nx = int((max_x - min_x) / x_step) + 1

min_y = -0.05
max_y = 0.05
y_step = 0.01
Ny = int((max_y - min_y) / y_step) + 1

min_z = -0.05
max_z = 0.05
z_step = 0.01
Nz = int((max_z - min_z) / z_step) + 1

xs = torch.tensor([min_x + x_step*i for i in range(Nx)] )
ys = torch.tensor([min_y + y_step*i for i in range(Ny)] )
zs = torch.tensor([min_z + z_step*i for i in range(Nz)] )

grid = torch.cartesian_prod(xs,ys,zs)

M = 100
board_positions = [BOARD_POSITIONS, 0.058, 0.108, 0.0875]
names = ['MSD', 'MAX XY', 'MAX Z', 'MAX Stiffness']

delta = 0.001
dx = create_points(1,1,delta,0,0)
dy = create_points(1,1,0,delta,0)
dz = create_points(1,1,0,0,delta)

Fxs = {}
Fys = {}
Fzs = {}
stiff = {}


for b_pos in board_positions:
    
    Fxs[b_pos] = []
    Fys[b_pos] = []
    Fzs[b_pos] = []
    stiff[b_pos] = []

    board = transducers(z=b_pos)

    for i,g in enumerate(grid):

        print(i,end='\r')

        p = create_points(1,1, *g)
        x = wgs(p,board=board)
        x = add_lev_sig(x)

        Fx1 = compute_force(x,p + dx,board=board)[0].detach().cpu()
        Fx2 = compute_force(x,p - dx,board=board)[0].detach().cpu()

        Fx = (Fx1 - Fx2) / (2*delta)
        Fxs[b_pos].append(Fx)

        Fy1 = compute_force(x,p + dy,board=board)[1].detach().cpu()
        Fy2 = compute_force(x,p - dy,board=board)[1].detach().cpu()

        Fy = (Fy1 - Fy2) / (2*delta)
        Fys[b_pos].append(Fy)

        Fz1 = compute_force(x,p + dz,board=board)[2].detach().cpu()
        Fz2 = compute_force(x,p - dz,board=board)[2].detach().cpu()
        
        Fz = (Fz1 - Fz2) / (2*delta)
        Fzs[b_pos].append(Fz)

        stiff[b_pos].append(-1*(Fx+Fy+Fz))

    print('\t\t',end='')



plt.subplot(2,2,1)
plt.boxplot(Fxs.values())
plt.xticks([1,2,3,4],labels=names)
plt.title('$\\nabla F_x$')

plt.subplot(2,2,2)
plt.boxplot(Fys.values())
plt.xticks([1,2,3,4],labels=names)
plt.title('$\\nabla F_y$')

plt.subplot(2,2,3)
plt.boxplot(Fzs.values())
plt.xticks([1,2,3,4],labels=names)
plt.title('$\\nabla F_z$')

plt.subplot(2,2,4)
plt.boxplot(stiff.values())
plt.xticks([1,2,3,4],labels=names)
plt.title('$-1 \\times (\\nabla F_x + \\nabla F_y +\\nabla F_z)$')

plt.show()