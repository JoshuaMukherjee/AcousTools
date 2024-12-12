from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Gorkov import gorkov_analytical
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Visualiser import Visualise_single_blocks, ABC

board = TRANSDUCERS

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch


# p = create_points(1,1,y=0, max_pos=0.04, min_pos=-0.04)
p = create_points(1,1,0,0,0)
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

F_trap_xs = []
F_trap_ys = []
F_trap_zs = []

F_twin_xs = []
F_twin_ys = []
F_twin_zs = []

F_vortex_xs = []
F_vortex_ys = []
F_vortex_zs = []

F_eye_xs = []
F_eye_ys = []
F_eye_zs = []

MAX_d = 0.015
N = 100
step = (MAX_d * 2) / N
pos = []

for i in range(N):
    d = -1 * MAX_d + i * step
    pd = p + create_points(1,1,d,0,0)
    pos.append(d)

    # F_trap_x, _, _ = compute_force(x_trap,pd,board, return_components=True)
    # F_twin_x, _, _ = compute_force(x_twin,pd,board, return_components=True)
    # F_vortex_x, _, _ = compute_force(x_vortex,pd,board, return_components=True)
    # F_eye_x, _, _ = compute_force(x_eye,pd,board, return_components=True)

    # F_trap_xs.append(F_trap_x.detach().cpu().item())
    # F_twin_xs.append(F_twin_x.detach().cpu().item())
    # F_vortex_xs.append(F_vortex_x.detach().cpu().item())
    # F_eye_xs.append(F_eye_x.detach().cpu().item())

    F_trap = force_fin_diff(x_trap, pd)
    F_twin = force_fin_diff(x_twin, pd)
    F_vortex = force_fin_diff(x_vortex, pd)
    F_eye = force_fin_diff(x_eye, pd)
    
    
    F_trap_xs.append(F_trap.squeeze()[0])
    F_twin_xs.append(F_twin.squeeze()[0])
    F_vortex_xs.append(F_vortex.squeeze()[0])
    F_eye_xs.append(F_eye.squeeze()[0])



for i in range(N):
    d = -1 * MAX_d + i * step
    pd = p + create_points(1,1,0,d,0)

    # _, F_trap_y, _ = compute_force(x_trap,pd,board, return_components=True)
    # _, F_twin_y, _= compute_force(x_twin,pd,board, return_components=True)
    # _, F_vortex_y, _ = compute_force(x_vortex,pd,board, return_components=True)
    # _, F_eye_y, _ = compute_force(x_eye,pd,board, return_components=True)

    # F_trap_ys.append(F_trap_y.detach().cpu().item())
    # F_twin_ys.append(F_twin_y.detach().cpu().item())
    # F_vortex_ys.append(F_vortex_y.detach().cpu().item())
    # F_eye_ys.append(F_eye_y.detach().cpu().item())

    F_trap = force_fin_diff(x_trap, pd)
    F_twin = force_fin_diff(x_twin, pd)
    F_vortex = force_fin_diff(x_vortex, pd)
    F_eye = force_fin_diff(x_eye, pd)
    
    
    F_trap_ys.append(F_trap.squeeze()[1])
    F_twin_ys.append(F_twin.squeeze()[1])
    F_vortex_ys.append(F_vortex.squeeze()[1])
    F_eye_ys.append(F_eye.squeeze()[1])


for i in range(N):
    d = -1 * MAX_d + i * step
    pd = p + create_points(1,1,0,0,d)

    # _, _,F_trap_z = compute_force(x_trap,pd,board, return_components=True)
    # _, _, F_twin_z = compute_force(x_twin,pd,board, return_components=True)
    # _, _, F_vortex_z = compute_force(x_vortex,pd,board, return_components=True)
    # _, _,F_eye_z = compute_force(x_eye,pd,board, return_components=True)

    # F_trap_zs.append(F_trap_z.detach().cpu().item())
    # F_twin_zs.append(F_twin_z.detach().cpu().item())
    # F_vortex_zs.append(F_vortex_z.detach().cpu().item())
    # F_eye_zs.append(F_eye_z.detach().cpu().item())


    F_trap = force_fin_diff(x_trap, pd)
    F_twin = force_fin_diff(x_twin, pd)
    F_vortex = force_fin_diff(x_vortex, pd)
    F_eye = force_fin_diff(x_eye, pd)
    
    
    F_trap_zs.append(F_trap.squeeze()[2])
    F_twin_zs.append(F_twin.squeeze()[2])
    F_vortex_zs.append(F_vortex.squeeze()[2])
    F_eye_zs.append(F_eye.squeeze()[2])



N = 5

plt.subplot(2,3,1)
plt.bar([1,],[p_trap,])
plt.bar([2,],[p_twin,])
plt.bar([3,],[p_vortex,])
plt.bar([4,],[p_eye,])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
plt.ylabel('Pressure (Pa)')

plt.subplot(2,3,2)
plt.bar([1,],[U_trap,])
plt.bar([2,],[U_twin,])
plt.bar([3,],[U_vortex,])
plt.bar([4,],[U_eye,])
plt.xticks([1,2,3,4],labels=labels,rotation=0)
# plt.yticks([i*1e-5 for i in range(5)], [-1*i*1e-5 for i in range(5)])
plt.ylabel('Gorkov')

plt.subplot(2,3,4)

plt.plot(pos,F_trap_xs)
plt.plot(pos,F_twin_xs)
plt.plot(pos,F_vortex_xs)
plt.plot(pos,F_eye_xs)
plt.ylabel('$F_x$ (N)')

plt.subplot(2,3,5)

plt.plot(pos,F_trap_ys)
plt.plot(pos,F_twin_ys)
plt.plot(pos,F_vortex_ys)
plt.plot(pos,F_eye_ys)
plt.ylabel('$F_y$ (N)')

plt.subplot(2,3,6)

plt.plot(pos,F_trap_zs)
plt.plot(pos,F_twin_zs)
plt.plot(pos,F_vortex_zs)
plt.plot(pos,F_eye_zs)
plt.ylabel('$F_z$ (N)')


# vmax = torch.max(torch.concat([img_trap,img_twin, img_vortex, img_eye]))
# vmin = torch.min(torch.concat([img_trap,img_twin, img_vortex, img_eye]))
# norm = mcolors.Normalize(vmin=vmin, vmax=vmax)


# ax = plt.subplot(3,4,9)
# im = plt.matshow(img_trap, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)


# ax = plt.subplot(3,4,10)
# plt.matshow(img_twin, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)

# ax = plt.subplot(3,4,11)
# plt.matshow(img_vortex, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)

# ax = plt.subplot(3,4,12)
# plt.matshow(img_eye, cmap='hot', fignum=0, norm=norm)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im,cax=cax)




# plt.tight_layout()
plt.show()