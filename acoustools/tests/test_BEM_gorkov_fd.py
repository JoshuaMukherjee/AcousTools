from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff
from acoustools.Solvers import wgs
from acoustools.BEM import propagate_BEM, BEM_gorkov_analytical, get_cache_or_compute_H, compute_E
from acoustools.Mesh import load_scatterer

import time
import scipy.stats
import matplotlib.pyplot as plt

board = TRANSDUCERS
path = "../BEMMedia"

USE_CACHE = True

sphere_pth =  path+"/Sphere-lam2.stl"
sphere = load_scatterer(sphere_pth, dy=-0.06, dz=0.03) #Make mesh at 0,0,3cm


N = 1000
points = []

H = get_cache_or_compute_H(sphere, board=board, path=path )

for i in range(N):
    p = create_points(1,1, max_pos=0.025)
    E = compute_E(sphere, p, board, H=H)
    x = wgs(p, board=board, A=E, iter=10)
    x = add_lev_sig(x)
    points.append([p,x])

U_fds = []
t_fds = []


U_as = []
t_as = []


for p,x in points:
    start_fd = time.time()
    U = gorkov_fin_diff(x,p, prop_function=propagate_BEM, prop_fun_args={'scatterer':sphere, 'path':path, "H":H} )
    end_fd = time.time()
    U_fds.append(-1*U.cpu().detach().item())
    t_fds.append(end_fd - start_fd)

    start_a = time.time()
    U = BEM_gorkov_analytical(x,p, scatterer=sphere, path=path,H=H)
    end_a = time.time()

    U_as.append(-1*U.cpu().detach().item())
    t_as.append(end_a - start_a)

# exit()

# print(end_fd - start_fd)
# print(end_a - start_a)

# print(U_fds)
# print(U_as)

plt.subplot(1,2,1)
plt.scatter(U_fds, U_as)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(' -U - Finite Differences')
plt.ylabel(' -U - Analytical')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(U_fds, U_as)
print(r_value**2)

plt.subplot(1,2,2)
plt.boxplot([t_fds,t_as])
plt.xticks([1,2],['Finite \n Differences', 'Analytical'])
plt.ylabel('Computation Time (s)')



plt.show()