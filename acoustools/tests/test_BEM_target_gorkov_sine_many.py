from acoustools.BEM import load_scatterer, BEM_gorkov_analytical, get_cache_or_compute_H
from acoustools.Utilities import create_points, TOP_BOARD, generate_gorkov_targets
from acoustools.Solvers import gradient_descent_solver
from acoustools.Visualiser import Visualise, ABC

from acoustools.Optimise.Constraints import sine_amplitude

import matplotlib.pyplot as plt


import vedo, torch, time
path = "../BEMMedia"

USE_CACHE = True
board = TOP_BOARD

reflector_path =  path+"/flat-lam2.stl"
reflector = load_scatterer(reflector_path, dz=0.0) #Make mesh at 0,0,0


H = get_cache_or_compute_H(reflector, board,path=path)
# E,F,G,H = compute_E(reflector,p,board, path=path, return_components=True)
# x = wgs(p,A=E)

# abc = ABC(0.07)
# Visualise(*abc, x,points=p, colour_functions=[propagate_BEM_pressure,],colour_function_args=[{'scatterer':reflector,'board':board, 'path':path, "H":H},])
def MSE_gorkov(transducer_phases, points, board, targets, **objective_params):
    transducer_phases = sine_amplitude(transducer_phases)
    U = BEM_gorkov_analytical(transducer_phases, points, reflector, board, path=path, H=H)
    loss = torch.mean((targets-U)**2).unsqueeze_(0).real
    return loss

N=1
M = 40

targets = {}
results = {}
times = {}

MAX= -4
MIN= -7

iters = [1,10,20,50,100]

for it in iters:
    targets[it] = []
    results[it] = []
    times[it] = []
    for i in range(M):
        
        U_targets = generate_gorkov_targets(N, max_val=MAX, min_val=MIN)
        p = create_points(N, min_pos=0.02, y=0)
        
        
        start = time.time_ns()
        x = gradient_descent_solver(p, MSE_gorkov, board, log=False, targets=U_targets, iters=it, lr=1e4, constrains=sine_amplitude)
        end = time.time_ns()
        Us = BEM_gorkov_analytical(x, p, reflector, board, path=path, H=H)
    
        for  U,U_target in zip(Us.squeeze(0,2), U_targets.squeeze(0,2)):
            print(i,U_target.item(),U.item(), U.item()/U_target.item())

            targets[it].append(U_target.item())
            results[it].append(U.item())
        times[it].append(end-start)


plt.subplot(2,1,1)
plt.title(f"N={N}")

for it in iters:
    plt.scatter([-i for i in targets[it]], [-i for i in results[it]], label=str(it))
plt.xlim(10**MIN,10**MAX)
plt.ylim(10**MIN,10**MAX)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("-1*U_tar")
plt.ylabel("-1*U_est")
plt.legend()

plt.subplot(2,1,2)
for i,it in enumerate(iters):
    mean_t = sum(times[it]) / len(times[it])
    plt.barh(i,mean_t)

plt.yticks(range(len(iters)), iters)
plt.ylabel('Iterations')
plt.xlabel('Time (ns)')


plt.show()