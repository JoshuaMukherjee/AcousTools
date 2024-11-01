from acoustools.Paths import interpolate_bezier
from acoustools.Utilities import create_points

import torch

import matplotlib.pyplot as plt


start =  create_points(1,1, z=0)
middle =    create_points(1,1, z=0)
end =    create_points(1,1,z=0)

offset_1 =  create_points(1,1, z=0) - start
offset_2 =  create_points(1,1, z=0) - start
offset_3 =  create_points(1,1, z=0) - middle

N =10

points = interpolate_bezier(start, middle, offset_1, offset_2,n=N)

points2 = interpolate_bezier(middle, end, middle - (start+offset_2), offset_3,n=N)



pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]

pts2 = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points2]


xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

xs2 = [p[0] for p in pts2]
ys2 = [p[1] for p in pts2]

start  = start.cpu().detach()
# middle = middle.cpu().detach()

plt.scatter(xs,ys)
plt.scatter(xs2,ys2)


plt.scatter(start[:,0],start[:,1],marker='x', label='Start')
plt.scatter((start+offset_1)[:,0],(start+offset_1)[:,1],marker='x', label='P2')
plt.scatter((start+offset_2)[:,0],(start+offset_2)[:,1],marker='x', label='P3')
plt.scatter(middle[:,0],middle[:,1],marker='x', label='Middle')
plt.scatter((start+offset_3)[:,0],(start+offset_3)[:,1],marker='x', label='P5')
plt.scatter(end[:,0],end[:,1],marker='x', label='End')


# plt.scatter(middle[:,0],middle[:,1],marker='x')

# plt.ylim((-0.05,0.05))
# plt.xlim((-0.05,0.05))
plt.legend()
plt.axis('equal')
plt.show()
