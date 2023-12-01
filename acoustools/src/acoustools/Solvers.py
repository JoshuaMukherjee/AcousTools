from acoustools.Utilities import *
import torch

def wgs(A, b, K):
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

def wgs_batch(A, b, iterations):
    AT = torch.conj(A).mT.to(device)
    b0 = b.to(device)
    x = torch.ones(A.shape[2],1).to(device) + 0j
    for kk in range(iterations):
        y = torch.matmul(A,x)                                   # forward propagate
        y = y/torch.max(torch.abs(y))                           # normalize forward propagated field (useful for next step's division)
        b = torch.multiply(b0,torch.divide(b,torch.abs(y)))     # update target - current target over normalized field
        b = b/torch.max(torch.abs(b))                           # normalize target
        p = torch.multiply(b,torch.divide(y,torch.abs(y)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram  
                    
    return y, p, x

def wgs_wrapper(points,iter = 200, board = TRANSDUCERS):
    A = forward_model_batched(points, board)
    _,_,act = wgs_batch(A,torch.ones(points.shape[2],1).to(device)+0j,iter)
    return act

def gspat(R,forward, backward, target, iterations):
    #Written by Giorgos Christopoulos 2022
    field = target 

    for _ in range(iterations):
        
#     amplitude constraint, keeps phase imposes desired amplitude ratio among points     
        target_field = torch.multiply(target,torch.divide(field,torch.abs(field)))  
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

def gspat_wrapper(points):
    A = forward_model_batched(points)
    backward = torch.conj(A).mT
    R = A@backward
    phase_hologram,pres = gspat(R,A,backward,torch.ones(points.shape[2],1).to(device)+0j, 200)
    return phase_hologram

def naive(points):
    activation = torch.ones(points.shape[1]) +0j
    activation = activation.to(device)
    forward = forward_model(points.T).to(device)
    back = torch.conj(forward).T
    # print(back.device, activation.device)
    trans = back@activation
    trans_phase=  trans / torch.abs(trans)
    out = forward@trans_phase
    pressure = torch.abs(out)
    return out, pressure

def naive_solver_batch(points,board=TRANSDUCERS):
    activation = torch.ones(points.shape[2],1) +0j
    activation = activation.to(device)
    forward = forward_model_batched(points,board)
    back = torch.conj(forward).mT
    trans = back@activation
    trans_phase=  trans / torch.abs(trans)
    out = forward@trans_phase

    return out, trans_phase

def naive_solver(points,transd=TRANSDUCERS):

    activation = torch.ones(points.shape[1]) +0j
    activation = activation.to(device)
    forward = forward_model(points,transd)
    back = torch.conj(forward).T
    trans = back@activation
    trans_phase=  trans / torch.abs(trans)
    out = forward@trans_phase


    return out, trans_phase

def naive_solver_wrapper(points):
    out,act = naive_solver_batch(points)
    return act

def ph_thresh(z_last,z,threshold):

    pi = torch.pi
    ph1 = torch.angle(z_last)
    ph2 = torch.angle(z)
    dph = ph2 - ph1
    
    dph = torch.atan2(torch.sin(dph),torch.cos(dph))    
    # print()
    # dph[dph>pi] = dph[dph>pi] - 2*pi
    # print((dph<-1*pi).any())
    # dph[dph<-1*pi] = dph[dph<-1*pi] + 2*pi
    # print((dph<-1*pi).any())
    
    dph[dph>threshold] = threshold
    dph[dph<-1*threshold] = -1*threshold
    
    # dph = torch.clamp(dph, -1*threshold, threshold)
    
    
    ph2 = ph1 + dph
    z = abs(z)*torch.exp(1j*ph2)
    
    return z

def soft(x,threshold):
    y = torch.max(torch.abs(x) - threshold,0).values
    y = y * torch.sign(x)
    return y

def ph_soft(x_last,x,threshold):
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

def temporal_wgs(A, y, K,ref_in, ref_out,T_in,T_out):
    '''
    Based off 
    Giorgos Christopoulos, Lei Gao, Diego Martinez Plasencia, Marta Betcke, 
    Ryuji Hirayama, and Sriram Subramanian. 2023. 
    Temporal acoustic point holography.(under submission) (2023)
    '''
    #ref_out -> points
    #ref_in-> transducers
    AT = torch.conj(A).T.to(device)
    y0 = y.to(device)
    x = torch.ones(A.shape[1],1).to(device) + 0j
    for kk in range(K):
        z = torch.matmul(A,x)                                   # forward propagate
        z = z/torch.max(torch.abs(z))                           # normalize forward propagated field (useful for next step's division)
        z = ph_thresh(ref_out,z,T_out); 
        
        y = torch.multiply(y0,torch.divide(y,torch.abs(z)))     # update target - current target over normalized field
        y = y/torch.max(torch.abs(y))                           # normalize target
        p = torch.multiply(y,torch.divide(z,torch.abs(z)))      # keep phase, apply target amplitude
        r = torch.matmul(AT,p)                                  # backward propagate
        x = torch.divide(r,torch.abs(r))                        # keep phase for hologram    
        x = ph_thresh(ref_in,x,T_in);    
    return y, p, x

def gradient_descent_solver(points, objective, board=TRANSDUCERS, optimiser=torch.optim.Adam, lr=0.01, 
                            objective_params={}, start=None, iters=200, 
                            maximise=False, targets=None, constrains=None, log=False):
    '''
    Solves phases using gradient descent\\
    `Objective` must take have an input of (`transducer_phases, points, board, targets, **objective_params`), `targets` may be `None` for unsupervised
    '''

    B = points.shape[0]
    N = points.shape[1]
    M = board.shape[0]
    if start is None:
        start = torch.ones((B,M,1)).to(device) +0j
    
    # param = torch.nn.Parameter(start).to(device)
    param = start.requires_grad_()
    optim = optimiser([param],lr)

    for epoch in range(iters):
        optim.zero_grad()       

        loss = objective(param, points, board, targets, **objective_params)
        
        if maximise:
            loss *= -1

        if log:
            print(epoch, loss)
        
        loss.backward()
        optim.step()
        
        if constrains is not None:
            param.data = constrains(param)

    return param
    
    



if __name__ == "__main__":
   points = create_points(4,2)
   x = naive_solver_wrapper(points)
   print(propagate_abs(x,points))

   x = wgs_wrapper(points)
   print(propagate_abs(x,points))
        