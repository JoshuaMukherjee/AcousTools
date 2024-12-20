from acoustools.Utilities import create_points, transducers, TRANSDUCERS, device, DTYPE, BOARD_POSITIONS, add_lev_sig
from acoustools.Solvers import gspat
from acoustools.Visualiser import Visualise_single_blocks, ABC, get_point_pos
from acoustools.Optimise.Objectives import gorkov_analytical_mean_objective
from acoustools.Optimise.Constraints import constrain_clamp_amp
from acoustools.Force import compute_force
from acoustools.Gorkov import gorkov_analytical
from acoustools.Stiffness import stiffness_finite_differences

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import pickle, math, torch

start = create_points(1,1,0,0,0.03)
origin = create_points(1,1,0,0,0)


# board = TRANSDUCERS
ITERATIONS = 20

xs = []
transducer_pos = []

abc = ABC(0.15)
N = 20
radius = 0.02


delta = 0.001
dx = create_points(1,1,delta,0,0)
dy = create_points(1,1,0,delta,0)
dz = create_points(1,1,0,0,delta)

def obj(param, points, board, targets, **objective_params):
    x = gspat(p, board=board, iterations=20)
    x = add_lev_sig(x)
    U = gorkov_analytical_mean_objective(x, points, board, targets)
    print(U)
    # F = compute_force(x,points,board)
 
    return torch.sum(U).unsqueeze(0),x 


def obj_stiff(param, points, board, targets, **objective_params):
    x = gspat(p, board=board, iterations=ITERATIONS)
    x = add_lev_sig(x)
    
    Fx1 = compute_force(x,points + dx,board=board)[0]
    Fx2 = compute_force(x,points - dx,board=board)[0]

    Fx = ((Fx1 - Fx2) / (2*delta))

    Fy1 = compute_force(x,points + dy,board=board)[1]
    Fy2 = compute_force(x,points - dy,board=board)[1]

    Fy = ((Fy1 - Fy2) / (2*delta))

    Fz1 = compute_force(x,points + dz,board=board)[2]
    Fz2 = compute_force(x,points - dz,board=board)[2]
    
    Fz = ((Fz1 - Fz2) / (2*delta))

    # F = compute_force(x,points,board)
 
    return (Fx + Fy + Fz).unsqueeze_(0),x 

def grad_transducer_solver(points, optimiser:torch.optim.Optimizer=torch.optim.Adam, 
                           lr=0.001, iters=1000, objective=None, targets=None, objective_params={},
                           log = True, maximise=False):


    B = points.shape[0]
    M = TRANSDUCERS.shape[0]
    
   
    board_height = torch.tensor([BOARD_POSITIONS,])+ torch.rand((1,),)/100
    board_height.requires_grad_() 
    # board_height = torch.rand((1,),)
    board_height.requires_grad_()
    optim = optimiser([board_height,],lr)


    for epoch in range(iters):
        optim.zero_grad()       
        board = transducers(z=board_height)
        loss,x = objective(None, points, board, targets, bh=board_height)
        # print(board_height)

        if log:
            print(epoch, loss.data)

        if maximise:
            loss *= -1
                
        
        loss.backward(torch.tensor([1]*B).to(device))
        optim.step()
    print(board_height)
    return x,board_height




compute = True
if compute:
    for i in range(N):
        print(i,end='\r')
        t = ((3.1415926*2) / N) * i
        x = radius * math.sin(t)
        z = radius * math.cos(t)
        
        p = create_points(1,1,x=x,y=0,z=z)
        
        x, board_height = grad_transducer_solver(p,objective=obj_stiff, log=False, iters=100, lr=0.01)        
        
        xs.append(x)

        transducer_pos.append(board_height)

    print()
    pickle.dump([xs,transducer_pos],open('imgs.pth','wb'))
else:
    xs,transducer_pos = pickle.load(open('imgs.pth','rb'))


fig,axs = plt.subplots(2,2)
res = (400,400)
ANIMATE = True

Us = []
stiffnesses = []

Us_static = []
stiffnesses_static = []


for i,x in enumerate(xs):

    b_h = transducer_pos[i]
    board = transducers(z=b_h)

    t = ((3.1415926*2) / N) * i
    x_pos = radius * math.sin(t)
    z_pos = radius * math.cos(t)
    
    p = create_points(1,1,x=x_pos,y=0,z=z_pos)
    
    stiffness = stiffness_finite_differences(x,p,board)
    stiffnesses.append(stiffness.cpu().detach().squeeze())

    U = gorkov_analytical(x,p,board).cpu().detach().squeeze()
    Us.append(U)

    x_static = gspat(p,TRANSDUCERS, iterations=ITERATIONS)
    x_static = add_lev_sig(x_static)
    
    stiffness_static = stiffness_finite_differences(x_static,p,TRANSDUCERS)
    stiffnesses_static.append(stiffness_static.cpu().detach().squeeze())
    
    U_static = gorkov_analytical(x_static,p,TRANSDUCERS).cpu().detach().squeeze()
    Us_static.append(U_static)



def traverse(index):
    
    print(index,end='\r')
    b_h = transducer_pos[index]
    print(index, b_h)

    x = xs[index]
    board = transducers(z=b_h)
    
    if ANIMATE:
        axs[0,0].clear()
        img = Visualise_single_blocks(*abc,x,res=res, colour_function_args={'board':board}).cpu().detach()
        im = axs[0,0].imshow(img,cmap='hot', vmax=9000)

        divider = make_axes_locatable(axs[0,0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


        b = create_points(1,1,0,0,b_h)
        pts_pos = get_point_pos(*abc,b,res)
        pts_pos_t = torch.stack(pts_pos).T
        axs[0][0].plot([10,res[0]-10],[pts_pos_t[0],pts_pos_t[0]],marker="x")

        b = create_points(1,1,0,0,-1*b_h)
        pts_pos = get_point_pos(*abc,b,res)
        pts_pos_t = torch.stack(pts_pos).T
        axs[0][0].plot([10,res[0]-10],[pts_pos_t[0],pts_pos_t[0]],marker="x")
    
    axs[0][1].clear()
    axs[0][1].plot(Us, label='Mobile')
    axs[0][1].plot(Us_static, label='Static')
    axs[0,1].legend(loc="upper right")

    axs[1][1].clear()
    axs[1][1].plot(stiffnesses, label='Mobile')
    axs[1][1].plot(stiffnesses_static, label='Static')
    axs[1,1].legend(loc="upper right")

        # plt.legend()

        
        

animation = animation.FuncAnimation(fig, traverse, frames=len(xs), interval=500)
    # plt.show()

animation.save('Results.gif', dpi=200, writer='imagemagick')