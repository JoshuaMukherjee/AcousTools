from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force
from acoustools.Visualiser import Visualise_single_blocks, ABC

board = TRANSDUCERS

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch


p = create_points(1,1,y=0, max_pos=0.04, min_pos=-0.04)
print(p)

x = wgs(p, board=board)

labels = ['Trap','Twin','Vortex','Eye' ]


x_trap = add_lev_sig(x, board, mode = 'Trap')
x_twin = add_lev_sig(x, board, mode = 'Twin')
x_vortex = add_lev_sig(x, board, mode = 'Vortex')
x_eye = add_lev_sig(x, board, mode = 'Eye')

A,B,C = ABC(0.06)

img_trap = Visualise_single_blocks(A,B,C,x_trap).cpu().detach()
img_twin = Visualise_single_blocks(A,B,C,x_twin).cpu().detach()
img_vortex = Visualise_single_blocks(A,B,C,x_vortex).cpu().detach()
img_eye = Visualise_single_blocks(A,B,C,x_eye).cpu().detach()



p_trap = propagate_abs(x_trap, p).cpu().detach().item()
p_twin = propagate_abs(x_twin, p).cpu().detach().item()
p_vortex = propagate_abs(x_vortex, p).cpu().detach().item()
p_eye = propagate_abs(x_eye, p).cpu().detach().item()


U_trap = gorkov_analytical(x_trap, p, board).cpu().detach().item()
U_twin = gorkov_analytical(x_twin, p, board).cpu().detach().item() 
U_vortex = gorkov_analytical(x_vortex, p, board).cpu().detach().item() 
U_eye= gorkov_analytical(x_eye, p, board).cpu().detach().item() 


F_trap_x, F_trap_y, F_trap_z = compute_force(x_trap,p,board, return_components=True)
F_twin_x, F_twin_y, F_twin_z = compute_force(x_twin,p,board, return_components=True)
F_vortex_x, F_vortex_y, F_vortex_z = compute_force(x_vortex,p,board, return_components=True)
F_eye_x, F_eye_y, F_eye_z = compute_force(x_eye,p,board, return_components=True)

N = 5

plt.subplot(3,4,1)
plt.bar([1,],[p_trap,])
plt.bar([2,],[p_twin,])
plt.bar([3,],[p_vortex,])
plt.bar([4,],[p_eye,])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
plt.ylabel('Pressure (Pa)')

plt.subplot(3,4,2)
plt.bar([1,],[U_trap,])
plt.bar([2,],[U_twin,])
plt.bar([3,],[U_vortex,])
plt.bar([4,],[U_eye,])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
# plt.yticks([i*1e-5 for i in range(5)], [-1*i*1e-5 for i in range(5)])
plt.ylabel('Gorkov')

plt.subplot(3,4,5)
plt.bar([1,],[F_trap_x.cpu().detach().item(),])
plt.bar([2,],[F_twin_x.cpu().detach().item(),])
plt.bar([3,],[F_vortex_x.cpu().detach().item(),])
plt.bar([4,],[F_eye_x.cpu().detach().item(),])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
# plt.yscale('log')
plt.ylabel('$F_x$ (N)')

plt.subplot(3,4,6)
plt.bar([1,],[F_trap_y.cpu().detach().item(),])
plt.bar([2,],[F_twin_y.cpu().detach().item(),])
plt.bar([3,],[F_vortex_y.cpu().detach().item(),])
plt.bar([4,],[F_eye_y.cpu().detach().item(),])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
# plt.yscale('log')
plt.ylabel('$F_y$ (N)')

plt.subplot(3,4,7)
plt.bar([1,],[F_trap_z.cpu().detach().item(),])
plt.bar([2,],[F_twin_z.cpu().detach().item(),])
plt.bar([3,],[F_vortex_z.cpu().detach().item(),])
plt.bar([4,],[F_eye_z.cpu().detach().item(),])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
# plt.yscale('log')
plt.ylabel('$F_z$ (N)')

vmax = torch.max(torch.concat([img_trap,img_twin, img_vortex, img_eye]))
vmin = torch.min(torch.concat([img_trap,img_twin, img_vortex, img_eye]))
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


ax = plt.subplot(3,4,9)
im = plt.matshow(img_trap, cmap='hot', fignum=0, norm=norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)


ax = plt.subplot(3,4,10)
plt.matshow(img_twin, cmap='hot', fignum=0, norm=norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)

ax = plt.subplot(3,4,11)
plt.matshow(img_vortex, cmap='hot', fignum=0, norm=norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)

ax = plt.subplot(3,4,12)
plt.matshow(img_eye, cmap='hot', fignum=0, norm=norm)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im,cax=cax)




# plt.tight_layout()
plt.show()