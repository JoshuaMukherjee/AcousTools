import torch
from torch import Tensor

def global_arnoldi(A:Tensor,k:int, s:int, V1=None):
    '''
    Algo. 2.1 in Global FOM and GMRES algorithms for matrix equations
    Could be replaced by 2.2 etc
    '''

    N = A.shape[1] #A is Batch x N x N

    Vs = torch.zeros(N,s,k) #To store for later

    if V1 is  None: V1 = torch.rand((N,s))
    V1 = V1 / torch.norm(V1, p='fro')
    Vs[:,:,0] = V1

    h = torch.zeros(k,k)


    for j in range(k-1):
        Vj = Vs[:,:,j]
        for i in range(j + 1):
            Vi = Vs[:,:,i]
            hij = Vi.mT @ A @ Vj
            hij = hij.squeeze(0)
            h[i,j] = torch.trace(hij) #Optimise this later -> can probably be vectorised

        sum_hv = 0
        for i in range(j+1):
            sum_hv = sum_hv + h[i,j] * Vs[:,:,i] #Optimise this later -> can probably be vectorised

        Vj_tilde = A.squeeze(0) @ Vj - sum_hv        
        Vj_tilde_norm = torch.norm(Vj_tilde, p='fro')

        h[j+1, j] = Vj_tilde_norm
        if Vj_tilde_norm == 0: return Vs

        Vs[:,:,j+1] = Vj_tilde / Vj_tilde_norm

    return Vs


def star_product(Vs, y):
    '''
    Eq. 2.1 in Global FOM and GMRES algorithms for matrix equations
    '''
    star_sum = 0
    for i in range(Vs.shape[2]):
        Vi = Vs[:,:,i]
        star_sum += Vi * y[i]
    return star_sum

def gi_gmres_solver(A, B, K):
    '''
    Algo. 4.1 in Global FOM and GMRES algorithms for matrix equations
    '''

    N = A.shape[1]
    s = B.shape[2]

    X0 = torch.rand(N,s)
    X = X0

    R = B - A@X
    V1 = R / torch.norm(R, p='fro')

    for k in range(1,K):

        Vs = global_arnoldi(A, k, s, V1)

        e1 = torch.eye(k+1)[0,:]

        H_tilde = torch.zeros((k+1, k)) #Defined above eq 2.1
        for i in range(k):
            for j in range(k):
                if i+1 >= j:
                    Vi = Vs[:,:,i]
                    Vj = Vs[:,:,j]

                    H_tilde[j,  i] = torch.trace(Vi.T @ A.squeeze(0) @ Vj) #Bottom corner == 0 -> Is it meant to?


        # print(e1)
        alpha = (torch.norm(R, p='fro') * e1)#.unsqueeze(1)
        y = torch.linalg.lstsq(H_tilde, alpha).solution
        # print(H_tilde.shape)
        # print(alpha.shape)
        y = (H_tilde.T @ H_tilde).inverse() @ H_tilde.T @ alpha #Is this the right way to do this?
        # print(y.shape)

        # exit()
#       
        # print(Vs.shape)
        # Vk = torch.reshape(Vs, (N, s*(k)))

        # Vk = torch.cat(Vs, dim=2)
        # print(k, X.shape, Vs.shape, y.shape)

        X = X + star_product(Vs, y)
        print((B - A@X).abs().sum())
        R = B - A@X

    return X









def gi_gmres(A, B, k):
    return gi_gmres_solver(A, B, k)