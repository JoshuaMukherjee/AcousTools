from svgpathtools import svg2paths
from acoustools.Paths import interpolate_bezier_velocity, svg_to_beziers, bezier_to_C1
from acoustools.Utilities import create_points

import matplotlib.pyplot as plt


pth = 'acoustools/tests/data/svgs/complex.svg'
points, bezier = svg_to_beziers(pth, True)
points, bezier = bezier_to_C1(bezier)


vels = []
for bez in bezier:
    vel = interpolate_bezier_velocity(*bez)
    vels += vel


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
plt.plot(xs,ys)



pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in vels]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]


plt.plot(xs,ys)
plt.show()