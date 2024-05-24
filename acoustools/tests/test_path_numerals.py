from acoustools.Paths import get_numeral

import torch
import matplotlib.pyplot as plt

A = torch.tensor((-0.02, 0.03,0))
B = torch.tensor((0.02, 0.03,0))
C = torch.tensor((-0.03, -0.02,0))


N= 9

fig = plt.figure()

for i in range(1,N+1):

    ax = fig.add_subplot(3,3,i)
    num = get_numeral(i,A,B,C)
    xs = [x[0] for x in num]
    ys = [x[1] for x in num]
    ax.plot(xs, ys)
    ax.set_xlim(A[0],B[0])
    ax.set_ylim(C[1],A[1])
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
