from svgpathtools import svg2paths
from acoustools.Paths import interpolate_bezier_velocity, svg_to_beziers, bezier_to_C1
from acoustools.Utilities import create_points

import matplotlib.pyplot as plt


pth = 'acoustools/tests/data/svgs/complex.svg'
points, spline = svg_to_beziers(pth, True)

points, spline = bezier_to_C1(spline)

vels = []
for bez in spline:
    vel = interpolate_bezier_velocity(bez)
    vels += vel


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
plt.plot(xs,ys, label='points')



pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in vels]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys, label='vels')

plt.legend()
plt.show()