from acoustools.Utilities import transducers, create_points, add_lev_sig, BOARD_POSITIONS
from acoustools.Solvers import wgs
from acoustools.Force import compute_force

import matplotlib.pyplot as plt


MSD_H = BOARD_POSITIONS

M = 400
board_positionns = [0.0005 * i for i in range(M)]

p = create_points(1,1,0,0,0)


delta = 0.001
dx = create_points(1,1,delta,0,0)
dy = create_points(1,1,0,delta,0)
dz = create_points(1,1,0,0,delta)

N = 16

Fxs = []
Fys = []

for i,b in enumerate(board_positionns):
    print(i,end='\r')


    board = transducers(N=N,z=b)
    x = wgs(p,board=board)
    x = add_lev_sig(x, board=board)

    Fx1 = compute_force(x,p + dx,board=board)[0]
    Fx2 = compute_force(x,p - dx,board=board)[0]

    Fx = ((Fx1 - Fx2) / (2*delta)).detach().cpu()

    Fy1 = compute_force(x,p + dy,board=board)[1]
    Fy2 = compute_force(x,p - dy,board=board)[1]

    Fy = ((Fy1 - Fy2) / (2*delta)).detach().cpu()

    Fz1 = compute_force(x,p + dz,board=board)[2]
    Fz2 = compute_force(x,p - dz,board=board)[2]
    
    Fz = ((Fz1 - Fz2) / (2*delta)).detach().cpu()
    
    Fxs.append(Fz/Fx)
    Fys.append(Fz/Fy)

print(min_fx:=min(Fxs), min_fx_i:=Fxs.index(min_fx), min_x:=board_positionns[min_fx_i])
print(min_fy:=min(Fys), min_fy_i:=Fys.index(min_fy), min_y:=board_positionns[min_fy_i])
# print(min_fz:=min(Fzs['Trap']), min_fz_i:=Fzs['Trap'].index(min_fz), min_z:=board_positionns[min_fz_i])

plt.subplot(1,2,1)

plt.plot(board_positionns, Fxs)
plt.ylabel('$\\nabla F_z / \\nabla F_x$')
plt.xlabel('+- Board Position (m)')
plt.axvline(MSD_H,linestyle='--', color='purple')
# plt.axvline(min_x,linestyle=':', color='red', label=f'Minima x, {min_x}m')
# plt.axvline(min_z,linestyle=':', color='blue', label=f'Minima z, {min_z}m')

plt.subplot(1,2,2)
plt.plot(board_positionns, Fys)
plt.ylabel('$\\nabla F_z / \\nabla F_y$')
plt.xlabel('+- Board Position (m)')
plt.axvline(MSD_H,linestyle='--', color='purple', label='MSD Height')
# plt.axvline(min_x,linestyle=':', color='red', label=f'Minima x, {min_x}m')
# plt.axvline(min_z,linestyle=':', color='blue', label=f'Minima z, {min_z}m')


# plt.tight_layout()
plt.show()
    