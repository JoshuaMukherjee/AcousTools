from acoustools.Paths import interpolate_arc
from acoustools.Utilities import create_points

import matplotlib.pyplot as plt

start =  create_points(1,1,x=0,y=0.01, z=0)
origin = create_points(1,1,x =  0,   y = 0.005,z=0)

points = interpolate_arc(start,origin=origin, anticlockwise=False, n=100)

pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]

xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

origin = origin.cpu().detach()
start  = start.cpu().detach()
# end = end.cpu().detach()

plt.scatter(xs,ys)
plt.scatter(origin[:,0],origin[:,1],marker='x')
plt.scatter(start[:,0],start[:,1],marker='x')
# plt.scatter(end[:,0],end[:,1],marker='x')

# plt.ylim((-0.05,0.05))
# plt.xlim((-0.05,0.05))
plt.axis('equal')
plt.show()