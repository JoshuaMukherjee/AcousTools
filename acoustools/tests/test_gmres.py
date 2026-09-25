from acoustools.linalg.gigmres import global_arnoldi, gi_gmres

import torch

N = 5
s = 7

A = torch.rand((N,N)) * 100 
B = torch.rand(1,N,s) * 100 

# A = torch.nn.init.sparse(A, sparsity=0.4).unsqueeze(0)
# B = torch.nn.init.sparse(B, sparsity=0.1)


Vs = global_arnoldi(A, 10, s)

# print(Vs)

for i in range(N):
    for j in range(N):
        Vi = Vs[:,:,i]
        Vj = Vs[:,:,j]
        if i!=j: assert(torch.trace(Vi.T @ Vj).abs() < 1e-4)
        if i==j: assert(torch.trace(Vi.T @ Vj).abs() > 0.95 and torch.trace(Vi.T @ Vj).abs() < 1.05)


X = gi_gmres(A,B,100, restart=None)

Xinv = A.inverse()@B

print(X / Xinv)

print(A@X - B)
print(A@Xinv - B)