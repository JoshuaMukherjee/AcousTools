from acoustools.linalg.gigmres import global_arnoldi, gi_gmres

import torch

N = 3
s = 2

A = torch.rand((1,N,N))
B = torch.rand(1,N,s)

Vs = global_arnoldi(A, 10, s)

# print(Vs)

for i in range(N):
    for j in range(N):
        Vi = Vs[:,:,i]
        Vj = Vs[:,:,j]
        if i!=j: assert(torch.trace(Vi.T @ Vj).abs() < 1e-4)
        if i==j: assert(torch.trace(Vi.T @ Vj).abs() > 0.95 and torch.trace(Vi.T @ Vj).abs() < 1.05)


X = gi_gmres(A,B,10)

Xinv = A.inverse()@B

print(X / Xinv)

print(A@X - B)