from acoustools.Levitator import LevitatorController
from acoustools.BEM import compute_E, propagate_BEM_pressure, BEM_gorkov_analytical
from acoustools.Mesh import load_scatterer
from acoustools.Utilities import create_points, TOP_BOARD, device, DTYPE
from acoustools.Solvers import gradient_descent_solver, naive, wgs
from acoustools.Optimise.Objectives import target_gorkov_BEM_mse_objective
from acoustools.Visualiser import ABC, Visualise_single
import acoustools.Constants as Constants

import pickle, time

import vedo, torch

import matplotlib.pyplot as plt

root = "../BEMMedia/" #Change to path to BEMMedia Folder
path = root+"flat-lam2.stl"

reflector = load_scatterer(path) #Change dz to be the position of the reflector


board = TOP_BOARD
U_target = torch.tensor([-7.5e-6,]).to(device).to(DTYPE)

B=1
N=1
X = 0.02
I = 100
p = create_points(1,1,0,0,0.05) #point at (0,0,0.05)
DX = 0.0001

x = naive(p, board)

xs = []
COMPUTE = False
if COMPUTE:
    for i in range(I):

        E,F,G,H = compute_E(reflector, p, board, path=root, return_components=True)

        new_points = p.expand(B,3,2*N).clone()
        SCALE = 2
        new_points[:,2,:N] -= Constants.wavelength / SCALE
        new_points[:,2,N:] += Constants.wavelength / SCALE
        target_phases = torch.zeros(B,2*N)
        target_phases[:,N:] = Constants.pi
        activation = torch.exp(1j * target_phases).unsqueeze(2).to(device)

        E2,F2,G2,H = compute_E(reflector, new_points, board, path=root, return_components=True, H=H)
        
    
        start = wgs(new_points,iter=2, board=board, return_components=False, A=E2)

        x = gradient_descent_solver(p, target_gorkov_BEM_mse_objective, board, log=False, targets=U_target, iters=100, 
                                    lr=1e5, init_type=start, objective_params={'reflector':reflector,'root':root,'dims':'Z'}, H=H)
        
        xs.append(x)

        pressure = propagate_BEM_pressure(x,p,reflector,E=E)
        U = BEM_gorkov_analytical(x, p, reflector, board, path=root).item()


        p[:,0] += DX


        print(i, pressure.item(), U, p[:,0].item())

    pickle.dump(xs,open('acoustools/tests/data/droplet' + str(I) + '.pth','wb'))
else:
    xs = pickle.load(open('acoustools/tests/data/droplet' + str(I) + '.pth','rb'))

abc = ABC(0.03, origin=(0,0,0.05))
for i,x in enumerate(xs):
    img = Visualise_single(*abc, x, colour_function=propagate_BEM_pressure, colour_function_args={'scatterer':reflector,'path':root})
    img = img.cpu().detach()
    plt.matshow(img, cmap='hot')
    plt.savefig('acoustools/tests/data/droplet/' + str(i) + '.png')


# lev = LevitatorController(ids=(73,)) #Change to your board IDs
# lev.set_frame_rate(200)
# lev.levitate(xs[0])
# input("Press Enter to move")
# for i,x in enumerate(xs):
#     lev.levitate(x)
    # input(f"{i}\r")
# input("Press Enter to end")
# lev.disconnect()