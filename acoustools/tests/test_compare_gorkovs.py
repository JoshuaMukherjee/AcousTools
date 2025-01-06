from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Gorkov import gorkov_analytical, gorkov_fin_diff
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

U_fds = []
t_fds = []
for p,x in points:
    start_fd = time.time()
    U = gorkov_fin_diff(x,p)
    end_fd = time.time()
    U_fds.append(U.cpu().detach().item())
    t_fds.append(end_fd - start_fd)

U_as = []
t_as = []

for p,x in points:
    start_a = time.time()
    U = gorkov_analytical(x,p)
    end_a = time.time()

    U_as.append(U.cpu().detach().item())
    t_as.append(end_a - start_a)


# print(end_fd - start_fd)
# print(end_a - start_a)

# print(U_fds)
# print(U_as)

plt.subplot(1,2,1)
plt.scatter(U_fds, U_as)
plt.xlabel(' U - Finite Differences')
plt.ylabel(' U - Analytical')

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(U_fds, U_as)
print(r_value**2)

plt.subplot(1,2,2)
plt.boxplot([t_fds,t_as])
plt.xticks([1,2],['Finite \n Differences', 'Analytical'])
plt.ylabel('Computation Time (s)')



plt.show()