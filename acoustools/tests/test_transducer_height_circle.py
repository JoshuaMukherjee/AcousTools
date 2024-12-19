from acoustools.Utilities import create_points, transducers, TRANSDUCERS, device, DTYPE, BOARD_POSITIONS
from acoustools.Solvers import wgs
from acoustools.Visualiser import Visualise_single_blocks, ABC, get_point_pos
from acoustools.Optimise.Objectives import gorkov_analytical_mean_objective
from acoustools.Optimise.Constraints import constrain_clamp_amp


import matplotlib.animation as animation
import matplotlib.pyplot as plt

import pickle, math, torch

start = create_points(1,1,0,0,0.03)
origin = create_points(1,1,0,0,0)


board = TRANSDUCERS

xs = []
transducer_pos = []

abc = ABC(0.15)
N = 30
radius = 0.02

def obj(param, points, board, targets, **objective_params):
    U = gorkov_analytical_mean_objective(param, points, board, targets)
    b_h = objective_params['bh']
    if b_h > 0.1 or b_h<0.03: 
        board_loss=100 
    else:
        board_loss=0
    # print(U, board_loss)
    return 1e3*U + board_loss

def grad_transducer_solver(points, optimiser:torch.optim.Optimizer=torch.optim.Adam, 
                           lr=0.001, iters=1000, objective=None, targets=None, objective_params={},
                           log = True, maximise=False):


    B = points.shape[0]
    M = TRANSDUCERS.shape[0]
    
    start = torch.e**(1j*torch.rand((B,M,1))*torch.pi)

    start=start.to(device).to(DTYPE)

    
    # param = torch.nn.Parameter(start).to(device)
    param = start.requires_grad_()
    board_height = torch.tensor([BOARD_POSITIONS,]).requires_grad_()
    optim = optimiser([param, board_height],lr)


    for epoch in range(iters):
        optim.zero_grad()       
        board = transducers(z=board_height)
        loss = objective(param, points, board, targets, bh=board_height)
        # print(board_height)

        if log:
            print(epoch, loss.data)

        if maximise:
            loss *= -1
                
        
        loss.backward(torch.tensor([1]*B).to(device))
        optim.step()
        param.data = constrain_clamp_amp(param)
    print(board_height)
    return param, board_height


res = (100,100)

compute = True
if compute:
    for i in range(N):
        print(i,end='\r')
        t = ((3.1415926*2) / N) * i
        x = radius * math.sin(t)
        z = radius * math.cos(t)
        
        p = create_points(1,1,x=x,y=0,z=z)
        
        x, board_height = grad_transducer_solver(p,objective=obj, log=False)        
        
        xs.append(x)

        transducer_pos.append(board_height)

    print()
    pickle.dump([xs,transducer_pos],open('imgs.pth','wb'))
else:
    xs,transducer_pos = pickle.load(open('imgs.pth','rb'))


fig = plt.figure()
ax = plt.gca()

def traverse(index):
    ax.clear()
    print(index,end='\r')
    b_h = transducer_pos[index]
    print(index, b_h)

    x = xs[index]
    board = transducers(z=b_h)
    img = Visualise_single_blocks(*abc,x,res=res, colour_function_args={'board':board}).cpu().detach()
    im = ax.imshow(img,cmap='hot', vmax=9000)

    b = create_points(1,1,0,0,b_h)
    pts_pos = get_point_pos(*abc,b,res)
    pts_pos_t = torch.stack(pts_pos).T
    ax.plot([10,res[0]-10],[pts_pos_t[0],pts_pos_t[0]],marker="x")

    b = create_points(1,1,0,0,-1*b_h)
    pts_pos = get_point_pos(*abc,b,res)
    pts_pos_t = torch.stack(pts_pos).T
    ax.plot([10,res[0]-10],[pts_pos_t[0],pts_pos_t[0]],marker="x")

    
    

animation = animation.FuncAnimation(fig, traverse, frames=len(xs), interval=500)
# plt.show()

animation.save('Results.gif', dpi=80, writer='imagemagick')