from acoustools.Utilities import transducers, create_points, add_lev_sig, BOARD_POSITIONS
from acoustools.Solvers import wgs
from acoustools.Force import compute_force

import matplotlib.pyplot as plt


MSD_H = BOARD_POSITIONS

M = 100
board_positionns = [0.002 * i for i in range(M)]

p = create_points(1,1,0,0,0)


delta = 0.001
dx = create_points(1,1,delta,0,0)
dy = create_points(1,1,0,delta,0)
dz = create_points(1,1,0,0,delta)

N = 16

Fxs = {}
Fys = {}
Fzs = {}

modes = ['Trap','Twin','Vortex','Eye']
for mode in modes:
    Fxs[mode] = []
    Fys[mode] = []
    Fzs[mode] = []

for i,b in enumerate(board_positionns):
    print(i,end='\r')
    for mode in modes:

        board = transducers(N=N,z=b)
        x = wgs(p,board=board)
        x = add_lev_sig(x,mode=mode, board=board)

        Fx1 = compute_force(x,p + dx,board=board)[0]
        Fx2 = compute_force(x,p - dx,board=board)[0]

        Fx = (Fx1 - Fx2) / (2*delta)
        Fxs[mode].append(Fx.detach().cpu())

        Fy1 = compute_force(x,p + dy,board=board)[1]
        Fy2 = compute_force(x,p - dy,board=board)[1]

        Fy = (Fy1 - Fy2) / (2*delta)
        Fys[mode].append(Fy.detach().cpu())

        Fz1 = compute_force(x,p + dz,board=board)[2]
        Fz2 = compute_force(x,p - dz,board=board)[2]
        
        Fz = (Fz1 - Fz2) / (2*delta)
        Fzs[mode].append(Fz.detach().cpu())

print(min_fx:=min(Fxs['Trap']), min_fx_i:=Fxs['Trap'].index(min_fx), min_x:=board_positionns[min_fx_i])
print(min_fy:=min(Fys['Trap']), min_fy_i:=Fys['Trap'].index(min_fy), min_y:=board_positionns[min_fy_i])
print(min_fz:=min(Fzs['Trap']), min_fz_i:=Fzs['Trap'].index(min_fz), min_z:=board_positionns[min_fz_i])

plt.subplot(1,3,1)
for mode in modes:
    plt.plot(board_positionns, Fxs[mode],label=mode)
plt.ylabel('$\\nabla F_x$')
plt.xlabel('+- Board Position (m)')
plt.axvline(MSD_H,linestyle='--', color='purple')
plt.axvline(min_x,linestyle=':', color='red', label=f'Minima x, {min_x}m')
plt.axvline(min_z,linestyle=':', color='blue', label=f'Minima z, {min_z}m')

plt.subplot(1,3,2)
for mode in modes:
    plt.plot(board_positionns, Fys[mode],label=mode)
plt.ylabel('$\\nabla F_y$')
plt.xlabel('+- Board Position (m)')
plt.axvline(MSD_H,linestyle='--', color='purple', label='MSD Height')
plt.axvline(min_x,linestyle=':', color='red', label=f'Minima x, {min_x}m')
plt.axvline(min_z,linestyle=':', color='blue', label=f'Minima z, {min_z}m')


plt.subplot(1,3,3)
for mode in modes:
    plt.plot(board_positionns, Fzs[mode],label=mode)
plt.ylabel('$\\nabla F_z$')
plt.xlabel('+- Board Position (m)')
plt.axvline(MSD_H,linestyle='--', color='purple')
plt.axvline(min_x,linestyle=':', color='red', label=f'Minima x, {min_x}m')
plt.axvline(min_z,linestyle=':', color='blue', label=f'Minima z, {min_z}m')
plt.legend()

# plt.tight_layout()
plt.show()
    