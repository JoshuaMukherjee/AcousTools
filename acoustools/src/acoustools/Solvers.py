from acoustools.Utilities import *
from acoustools.Optimise.Constraints import constrain_phase_only
from acoustools.Constraints import constrain_amplitude, constrain_field, constrain_field_weighted
import torch

from torch import Tensor
from types import FunctionType

def wgs_solver_unbatched(A, b, K):
    '''
    @private
    unbatched WGS solver for transducer phases, better to use `wgs_solver_batch` \\
    `A` Forward model matrix to use \\ 
    `b` initial guess - normally use `torch.ones(N,1).to(device)+0j`\\
    `k` number of iterations to run for \\
    returns (hologram image, point phases, hologram)
    '''
    #Written by Giorgos Christopoulos 2022
    AT = torch.conj(A).T.to(device)
    b0 = b.to(device)
    x = torch.ones(A.shape[1],1).to(device) + 0j
    for kk in range(K):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return y, p, x

def wgs_solver_batch(A, b, iterations):
    '''
    @private
    batched WGS solver for transducer phases\\
    `A` Forward model matrix to use \\ 
    `b` initial guess - normally use `torch.ones(self.N,1).to(device)+0j`\\
    `iterations` number of iterations to run for \\
    returns (point pressure ,point phases, hologram)
    '''
    AT = torch.conj(A).mT.to(device)
    
    b = b.expand(A.shape[0],-1,-1)
    b0 = b.to(device).expand(A.shape[0],-1,-1)
    
    x = torch.ones(A.shape[2],1).to(device) + 0j
    for kk in range(iterations):
        p = A@x
        p,b = constrain_field_weighted(p,b0,b)
        x = AT@p
        x = constrain_amplitude(x)
    y =  torch.abs(A@x) 
    return y, p, x

def wgs(points:Tensor,iter:int = 200, board:Tensor|None = None, A:Tensor|None = None, b:Tensor|None=None, return_components:bool=False) -> Tensor:
    '''
    Weighted-GS algorithm\n
    :param points: Points to use
    :param iter: Number of iterations for WGS, default:`200`
    :param board: The Transducer array, If `None` uses default two 16x16 arrays
    :param A: Forward model matrix to use 
    :param b: initial guess - If none will use `torch.ones(N,1).to(device)+0j`
    :param return_components: IF True will return `hologram image, point phases, hologram` else will return `hologram`, default False
    :return: hologram

    ```Python
    from acoustools.Solvers import wgs
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(2,4)
    print(p)
    x = wgs(p)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = wgs(p)
    print(propagate_abs(x,p))
    ```
    '''
    if board is None:
        board = TRANSDUCERS

    if len(points.shape) > 2:
        N = points.shape[2]
        batch=True
    else:
        N = points.shape[1]
        batch=False

    if A is None:
        A = forward_model(points, board)
    if b is None:
        b = torch.ones(N,1).to(device)+0j

    if batch:
        img,pha,act = wgs_solver_batch(A,b,iter)
    else:
        img,pha,act = wgs_solver_unbatched(A,b,iter)

    if return_components:
        return img,pha,act
    return act

def gspat_solver(R,forward, backward, target, iterations):
    '''
    @private
    GS-PAT Solver for transducer phases\\
    `R` R Matrix\\
    `forward` forward propagation matrix\\
    `backward` backward propagation matrix\\
    `target` initial guess - can use `torch.ones(N,1).to(device)+0j`
    `iterations` Number of iterations to use\\
    returns (hologram, point activations)
    '''
    #Written by Giorgos Christopoulos 2022
    field = target 

    for _ in range(iterations):
        
#     amplitude constraint, keeps phase imposes desired amplitude ratio among points     
        target_field = constrain_field(field, target)
#     backward and forward propagation at once
        field = torch.matmul(R,target_field)
#     AFTER THE LOOP
#     impose amplitude constraint and keep phase, after the iterative part this step is different following Dieg
    target_field = torch.multiply(target**2,torch.divide(field,torch.abs(field)**2))
#     back propagate 
    complex_hologram = torch.matmul(backward,target_field)
#     keep phase 
    phase_hologram = torch.divide(complex_hologram,torch.abs(complex_hologram))
    points = torch.matmul(forward,phase_hologram)

    return phase_hologram, points

def gspat(points:Tensor|None=None, board:Tensor|None=None,A:Tensor|None=None,B:Tensor|None=None, 
          R:Tensor|None=None ,b:Tensor|None = None, iterations:int=200, return_components:bool=False) -> Tensor:
    '''
    GSPAT Solver\n
    :param points: Target point positions
    :param board: The Transducer array, if None uses default two 16x16 arrays
    :param A: The Forward propagation matrix, if `None` will be computed 
    :param B: The backwards propagation matrix, if `None` will be computed 
    :param R: The R propagation matrix, if `None` will be computed 
    :param b: initial guess - If None will use `torch.ones(N,1).to(device)+0j`
    :param iterations: Number of iterations to use
    :param return_components: IF True will return `hologram, pressure` else will return `hologram`, default True
    :return: Hologram
    
    ```Python
    from acoustools.Solvers import wgs
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(1,1)
    print(p)
    x = wgs(p)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = wgs(p)
    print(propagate_abs(x,p))
    ```
    '''

    if board is None:
        board = TRANSDUCERS

    if A is None:
        A = forward_model(points,board)
    if B is None:
        B = torch.conj(A).mT
    if R is None:
        R = A@B

    if b is None:
        if is_batched_points(points):
            b = torch.ones(points.shape[2],1).to(device)+0j
        else:
            b = torch.ones(points.shape[1],1).to(device)+0j
    phase_hologram,pres = gspat_solver(R,A,B,b, iterations)
    
    if return_components:
        return phase_hologram,pres
    return phase_hologram


def naive_solver_batched(points,board=TRANSDUCERS, activation=None):
    '''
    @private
    Batched naive (backpropagation) algorithm for phase retrieval\\
    `points` Target point positions\\
    `board` The Transducer array, default two 16x16 arrays\\
    `activation` Initial starting point activation \\
    returns (point activations, hologram)
    '''
    if activation is None:
        activation = torch.ones(points.shape[2],1) +0j
        activation = activation.to(device)
    forward = forward_model_batched(points,board)
    back = torch.conj(forward).mT
    trans = back@activation
    trans_phase=  constrain_amplitude(trans)
    out = forward@trans_phase

    return out, trans_phase

def naive_solver_unbatched(points,board=TRANSDUCERS, activation=None):
    '''
    @private
    Unbatched naive (backpropagation) algorithm for phase retrieval\\
    `points` Target point positions\\
    `board` The Transducer array, default two 16x16 arrays\\
    `activation` Initial starting point activation \\
    returns (point activations, hologram)
    '''
    if activation is None:
        activation = torch.ones(points.shape[1]) +0j
        activation = activation.to(device)
    forward = forward_model(points,board)
    back = torch.conj(forward).T
    trans = back@activation
    trans_phase=  constrain_amplitude(trans)
    out = forward@trans_phase


    return out, trans_phase

def naive(points:Tensor, board:Tensor|None = None, return_components:bool=False, activation:Tensor|None=None) -> Tensor:
    '''
    Naive solver\n
    :param points: Target point positions
    :param board: The Transducer array, default two 16x16 arrays
    :param return_components: If `True` will return `hologram, pressure` else will return `hologram`, default True
    :param activation: Initial starting point activation 
    :return: hologram
    '''
    if board is None:
        board = TRANSDUCERS
    if is_batched_points(points):
        out,act = naive_solver_batched(points,board=board, activation=activation)
    else:
        out,act = naive_solver_unbatched(points,board=board, activation=activation)
    if return_components:
        return act, out
    return act

def ph_thresh(z_last,z,threshold):
    '''
    @private
    Phase threshhold between two timesteps point phases, clamps phase changes above `threshold` to be `threshold`\\
    `z_last` point activation at timestep t-1\\
    `z` point activation at timestep t\\
    `threshold` maximum allowed phase change\\
    returns constrained point activations
    '''

    ph1 = torch.angle(z_last)
    ph2 = torch.angle(z)
    dph = ph2 - ph1
    
    dph = torch.atan2(torch.sin(dph),torch.cos(dph)) 
    
    dph[dph>threshold] = threshold
    dph[dph<-1*threshold] = -1*threshold
    
    ph2 = ph1 + dph
    z = abs(z)*torch.exp(1j*ph2)
    
    return z

def soft(x,threshold):
    '''
    @private
    Soft threshold for a set of phase changes, will return the change - threshold if change > threshold else 0\\
    `x` phase changes\\
    `threshold` Maximum allowed hologram phase change\\
    returns new phase changes
    '''
    y = torch.max(torch.abs(x) - threshold,0).values
    y = y * torch.sign(x)
    return y

def ph_soft(x_last,x,threshold):
    '''
    @private
    Soft thresholding for holograms \\
    `x_last` Hologram from timestep t-1\\
    `x` Hologram from timestep t \\
    `threshold` Maximum allowed phase change\\
    returns constrained hologram
    '''
    pi = torch.pi
    ph1 = torch.angle(x_last)
    ph2 = torch.angle(x)
    dph = ph2 - ph1

    dph[dph>pi] = dph[dph>pi] - 2*pi
    dph[dph<-1*pi] = dph[dph<-1*pi] + 2*pi

    dph = soft(dph,threshold)
    ph2 = ph1 + dph
    x = abs(x)*torch.exp(1j*ph2)
    return x

def temporal_wgs(A:Tensor, y:Tensor, K:int,ref_in:Tensor, ref_out:Tensor,T_in:float,T_out:float) -> Tensor:
    '''
    Based off `
    Giorgos Christopoulos, Lei Gao, Diego Martinez Plasencia, Marta Betcke, 
    Ryuji Hirayama, and Sriram Subramanian. 2023. 
    Temporal acoustic point holography. (2024) https://doi.org/10.1145/3641519.3657443 \n
    WGS solver for hologram where the phase change between frames is constrained \n
    :param A: Forward model  to use
    :param y: initial guess to use normally use `torch.ones(self.N,1).to(device)+0j`
    :param K: Number of iterations to use
    :param ref_in: Previous timesteps hologram
    :param ref_out: Previous timesteps point activations
    :param T_in: Hologram phase change threshold
    :param T_out: Point activations phase change threshold
    :return: (hologram image, point phases, hologram)
    

    '''
    #ref_out -> points
    #ref_in-> transducers
    AT = torch.conj(A).mT.to(device)
    y0 = y.to(device)
    x = torch.ones(A.shape[2],1).to(device) + 0j
    for kk in range(K):
        z = torch.matmul(A,x)                                   # forward propagate
        z = z/torch.max(torch.abs(z),dim=1,keepdim=True).values # normalize forward propagated field (useful for next step's division)
        z = ph_thresh(ref_out,z,T_out); 
        
        y = torch.multiply(y0,torch.divide(y,torch.abs(z)))     # update target - current target over normalized field
        y = y/torch.max(torch.abs(y),dim=1,keepdim=True).values # normalize target
        p = torch.multiply(y,torch.divide(z,torch.abs(z)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram    
        x = ph_thresh(ref_in,x,T_in);    
    return y, p, x






def gradient_descent_solver(points: Tensor, objective: FunctionType, board:Tensor|None=None, optimiser:torch.optim.Optimizer=torch.optim.Adam, lr: float=0.01, 
                            objective_params:dict={}, start:Tensor|None=None, iters:int=200, 
                            maximise:bool=False, targets:Tensor=None, constrains:FunctionType=constrain_phase_only, log:bool=False, return_loss:bool=False,
                            scheduler:torch.optim.lr_scheduler.LRScheduler=None, scheduler_args:dict=None, save_each_n:int = 0, save_set_n:list[int] = None) -> Tensor:
    '''
    Solves phases using gradient descent\n
    :param points: Target point positions 
    :param objective: Objective function - must take have an input of (`transducer_phases, points, board, targets, **objective_params`), `targets` may be `None` for unsupervised
    :param board: The Transducer array, default two 16x16 arrays
    :param optimiser: Optimiser to use (should be compatable with the interface from from `torch.optim`). Default `torch.optim.Adam`
    :param lr: Learning Rate to use. Default `0.01`
    :param objective_params: Any parameters to be passed to `objective` as a dictionary of `{parameter_name:parameter_value}` pairs. Default `{}`
    :param start: Initial guess. If None will default to a random initilsation of phases 
    :param iters: Number of optimisation Iterations. Default 200
    :param maximise: Set to `True` to maximise the objective, else minimise. Default `False`
    :param targets: Targets to optimise towards for supervised optimisation, unsupervised if set to `None`. Default `None`
    :param constrains: Constraints to apply to result 
    :param log: If `True` prints the objective values at each step. Default `False`
    :param return_loss: If `True` save and return objective values for each step as well as the optimised result 
    :param scheduler: Learning rate scheduler to use, if `None` no scheduler is used. Default `None` 
    :param scheduler_args: Parameters to pass to `scheduler`
    :param save_each_n: For n>0 will save the optimiser results at every n steps. Set either `save_each_n` or `save_set_iters`
    :param save_set_iters: List containing exact iterations to save optimiser results at. Set either `save_each_n` or `save_set_iters`
    :return: optimised result and optionally the objective values and results (see `return_loss`, `save_each_n` and `save_set_iters`). If either are returned both will be returned but maybe empty if not asked for
    
    ```Python
    from acoustools.Optimise.Objectives import propagate_abs_sum_objective
    from acoustools.Solvers import gradient_descent_solver
    from acoustools.Optimise.Constraints import constrain_phase_only

    p = create_points(4,2)
    x = gradient_descent_solver(p,propagate_abs_sum_objective, 
                                maximise=True, constrains=constrain_phase_only, 
                                log=False, lr=1e-1)

    print(propagate_abs(x,p))

    ```
    ''' 

    if board is None:
        board = TRANSDUCERS

    losses = []
    results = {}
    B = points.shape[0]
    N = points.shape[1]
    M = board.shape[0]
    if start is None:
        # start = torch.ones((B,M,1)).to(device) +0j
        start = torch.e**(1j*torch.rand((B,M,1))*torch.pi)

        start=start.to(device).to(DTYPE)
    
    
    # param = torch.nn.Parameter(start).to(device)
    param = start.requires_grad_()
    optim = optimiser([param],lr)
    if scheduler is not None:
        scheduler = scheduler(optim,**scheduler_args)


    for epoch in range(iters):
        optim.zero_grad()       

        loss = objective(param, points, board, targets, **objective_params)

        if log:
            print(epoch, loss.data)

        if maximise:
            loss *= -1
        
        if return_loss:
            losses.append(loss)
        
        if save_each_n > 0 and epoch % save_each_n == 0:
            results[epoch] = param.clone().detach()
        elif save_set_n is not None and epoch in save_set_n:
            results[epoch] = param.clone().detach()

        
        loss.backward(torch.tensor([1]*B).to(device))
        optim.step()
        if scheduler is not None:
            scheduler.step()
        
        if constrains is not None:
            param.data = constrains(param)

    if return_loss or save_each_n > 0:
        return param, losses, results
    

    return param
    

def iterative_backpropagation(points:Tensor,iterations:int = 200, board:Tensor|None = None, A:Tensor|None = None, b:Tensor|None=None, return_components: bool=False) -> list[Tensor]:
    '''
    IB solver for transducer phases\n
    :param points: Points to use
    :param iterations: Number of iterations for WGS, default`200`
    :param board: The Transducer array, default two 16x16 arrays
    :param A: Forward model matrix to use 
    :param b: initial guess - If none will use `torch.ones(N,1).to(device)+0j`
    :param return_components: IF True will return `hologram image, point phases, hologram` else will return `hologram`, default False
    :return: (point pressure ,point phases, hologram)

    ```Python
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(2,1)
    print(p)
    x = iterative_backpropagation(p)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = iterative_backpropagation(p)
    print(propagate_abs(x,p))

    ```
    '''

    if board is None:
        board  = TRANSDUCERS
    
    if len(points.shape) > 2:
        N = points.shape[2]
        batch=True
    else:
        N = points.shape[1]
        batch=False

    if A is None:
        A = forward_model(points, board)
    
    if batch:
        M = A.shape[2]
    else:
        M = A.shape[1]


    if b is None:
        b = torch.ones(N,1).to(device)+0j
    
    AT =  torch.conj(A).mT.to(device)
    x = torch.ones(M,1).to(device) + 0j
    for kk in range(iterations):
        p = A@x
        p = constrain_field(p,b)
        x = AT@p
        x = constrain_amplitude(x)
    y =  torch.abs(A@x) 
    if return_components:
        return y, p, x
    else:
        return x