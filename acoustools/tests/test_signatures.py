from acoustools.Utilities import create_points, add_lev_sig, propagate_abs, TRANSDUCERS, BOTTOM_BOARD
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise_single

import torch

import matplotlib.pyplot as plt

board = TRANSDUCERS

p = create_points(1,x=0,y=0,z=0)
x = wgs(p, board=board)




plane = board[:,0:2]
sig = torch.atan2(plane[:,0], plane[:,1]).unsqueeze(0).unsqueeze(2)
sig_twin = torch.zeros_like(sig) + torch.pi * (plane[:,0] > 0).unsqueeze(0).unsqueeze(2)

# sig  = torch.atan(plane[:,1]/plane[:,0]).unsqueeze(0).unsqueeze(2)


focal_x = x.clone()
x = torch.abs(x) * torch.exp(1j* (torch.angle(x) + sig))
x_twin = torch.abs(focal_x) * torch.exp(1j* (torch.angle(focal_x) + sig_twin))


A = torch.tensor((-0.02, 0.02,0))
B = torch.tensor((0.02, 0.02,0))
C = torch.tensor((-0.02, -0.02,0))

# A = torch.tensor((-0.02,0, 0.02))
# B = torch.tensor((0.02,0, 0.02))
# C = torch.tensor((-0.02,0, -0.02))

normal = (0,1,0)
origin = (0,0,0)

fig = plt.figure()

ax1= fig.add_subplot(3,3,1)
ax1.set_title("Vortex")
ax1.set_ylabel("Pressure")
img = Visualise_single(A,B,C, x, colour_function_args={"board":board})
img_p = ax1.matshow(img, cmap='hot')
fig.colorbar(img_p, ax=ax1)



ax2 = fig.add_subplot(3,3,2)
# sig_im = ax2.scatter(plane[:,0],plane[:,1],c=sig, cmap='hsv')
ax2.set_title("Focal Point")
img = Visualise_single(A,B,C, focal_x, colour_function_args={"board":board})
img_p_focal = ax2.matshow(img, cmap='hot')
fig.colorbar(img_p_focal, ax=ax2)

ax7 = fig.add_subplot(3,3,3)
# sig_im = ax2.scatter(plane[:,0],plane[:,1],c=sig, cmap='hsv')
ax7.set_title("Twin Trap")
img = Visualise_single(A,B,C, x_twin, colour_function_args={"board":board})
img_p_twin = ax7.matshow(img, cmap='hot')
fig.colorbar(img_p_twin, ax=ax7)


ax3 = fig.add_subplot(3,3,4)
ax3.set_ylabel("Signature")
sig_im_vortex = ax3.matshow(sig.real.reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax3.set_aspect('equal', adjustable='box')
fig.colorbar(sig_im_vortex, ax=ax3)


ax4 = fig.add_subplot(3,3,5)
sig_im_focal = ax4.matshow(torch.zeros_like(sig).reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax4.set_aspect('equal', adjustable='box')
fig.colorbar(sig_im_focal, ax=ax4)

ax8 = fig.add_subplot(3,3,6)
sig_im_twin = ax8.matshow(sig_twin.real.reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax8.set_aspect('equal', adjustable='box')
fig.colorbar(sig_im_twin, ax=ax8)


ax5 = fig.add_subplot(3,3,7)
ax5.set_ylabel("Hologram")
holo_im_vortex = ax5.matshow(torch.angle(x).reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax5.set_aspect('equal', adjustable='box')
fig.colorbar(holo_im_vortex, ax=ax5)

ax6 = fig.add_subplot(3,3,8)
holo_im_focal = ax6.matshow(torch.angle(focal_x).reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax6.set_aspect('equal', adjustable='box')
fig.colorbar(holo_im_focal, ax=ax6)

ax9 = fig.add_subplot(3,3,9)
holo_im_twin = ax9.matshow(torch.angle(x_twin).reshape(1,-1,16,16)[0,0,:,:].mT, cmap='hsv', vmin=-3.14159, vmax=3.14159)
ax9.set_aspect('equal', adjustable='box')
fig.colorbar(holo_im_focal, ax=ax9)




plt.show()