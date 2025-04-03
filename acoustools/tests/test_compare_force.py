from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Force import compute_force, force_fin_diff
from acoustools.Solvers import wgs

import time
import scipy.stats
import matplotlib.pyplot as plt

board = TRANSDUCERS

N = 1000
points = []


for i in range(N):
    p = create_points(1,1)
    x = wgs(p, board=board)
    x = add_lev_sig(x)
    points.append([p,x])

Fx_fds = []
Fy_fds = []
Fz_fds = []


t_fds = []
for p,x in points:
    start_fd = time.time()
    Fx,Fy,Fz = force_fin_diff(x,p).squeeze_()
    end_fd = time.time()
    Fx_fds.append(Fx.cpu().detach().item())
    Fy_fds.append(Fy.cpu().detach().item())
    Fz_fds.append(Fz.cpu().detach().item())
    t_fds.append(end_fd - start_fd)


Fx_as = []
Fy_as = []
Fz_as = []

t_as = []

for p,x in points:
    start_a = time.time()
    Fx,Fy,Fz = compute_force(x,p, return_components=True)
    print(Fz)
    end_a = time.time()

    Fx_as.append(Fx.cpu().detach().item())
    Fy_as.append(Fy.cpu().detach().item())
    Fz_as.append(Fz.cpu().detach().item())

    t_as.append(end_a - start_a)


# print(end_fd - start_fd)
# print(end_a - start_a)

# print(U_fds)
# print(U_as)

plt.subplot(3,2,1)
plt.scatter(Fx_fds, Fx_as, label='$F_x$', color='blue')
plt.xlabel('$F_x$ - Finite Differences')
plt.ylabel('$F_x$ - Analytical')


plt.subplot(3,2,3)
plt.scatter(Fy_fds, Fy_as, label='$F_y$', color='orange')
plt.xlabel('$F_y$ - Finite Differences')
plt.ylabel('$F_x$ - Analytical')


plt.subplot(3,2,5)
plt.scatter(Fz_fds, Fz_as, label='$F_z$', color='green')
plt.xlabel(' $F_z$ - Finite Differences')
plt.ylabel(' $F_z$ - Analytical')

slope, intercept, r_value_x, p_value, std_err = scipy.stats.linregress(Fx_fds, Fx_as)
slope, intercept, r_value_y, p_value, std_err = scipy.stats.linregress(Fy_fds, Fy_as)
slope, intercept, r_value_z, p_value, std_err = scipy.stats.linregress(Fz_fds, Fz_as)
print(r_value_x**2, r_value_y**2, r_value_z**2)

plt.subplot(1,2,2)
plt.boxplot([t_fds,t_as])
plt.xticks([1,2],['Finite \n Differences', 'Analytical'])
plt.ylabel('Computation Time (s)')



plt.show()