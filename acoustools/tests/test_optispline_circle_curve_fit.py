from acoustools.Paths.Bezier import create_bezier_circle, interpolate_bezier, close_bezier
from acoustools.Paths.OptiSpline import OptiSpline
from acoustools.Optimise.OptiSpline_Objectives import optispline_min_distance_control_points
from acoustools.Paths import interpolate_arc
from acoustools.Utilities import create_points

import matplotlib.pyplot as plt

import copy,math

import matplotlib.animation as animation
import matplotlib  as mpl



N = 20 #Number of curves to start with
n = N*3 #Number of samples in total
spline_circle = create_bezier_circle(N-1)
spline_circle = close_bezier(spline_circle,0)
c = copy.deepcopy(spline_circle)

start =  create_points(1,1,x=0,y=0.03, z=0)
origin = create_points(1,1,x =0,y=0,z=0)
target_circle = interpolate_arc(start,origin=origin, n=n)

# target_circle = [p+create_points(1,1, max_pos=0.002, min_pos=-0.002) for i,p in enumerate(target_circle)]  

target_circle = [p+create_points(1,1, x=math.sin(6*i/6.28)/100, y=math.sin(6*i/6.28)/100, z=0 ) for i,p in enumerate(target_circle)]  


spline, saved = OptiSpline(spline_circle,target_circle,optispline_min_distance_control_points,n=int(n/N), 
                    optimise_start=True, C1=True, get_intermediates=True, iters=100, lr=0.001)


fig = plt.figure()
fig.clear()

circle = []
for bez in c:
    circle += interpolate_bezier(bez,20)
xs = [pt[:,0].item() for pt in circle]
ys = [pt[:,1].item() for pt in circle]


xs_target = [pt[:,0].item() for pt in target_circle]
ys_target = [pt[:,1].item() for pt in target_circle]

def traverse(index):

    fig.clear()

    result = []
    for bez in saved[index]:
        result += interpolate_bezier(bez,20)

    
    xs_result = [pt[:,0].item() for pt in result]
    ys_result = [pt[:,1].item() for pt in result]

    plt.scatter(xs,ys, label='Start', marker='.')
    plt.plot(xs_target,ys_target, label='Target', marker='x')
    plt.plot(xs_result,ys_result, label='Result', marker='.', color='red')

    plt.xlim(-0.05,0.05)
    plt.ylim(-0.05,0.05)
    plt.legend()



ani = animation.FuncAnimation(fig, traverse, frames=len(saved), interval=50)

ani.save("OptiSpline.gif", writer='imagemagick', dpi = 200)