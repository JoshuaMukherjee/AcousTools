from svgpathtools import svg2paths
from acoustools.Paths import interpolate_bezier, svg_to_beziers
from acoustools.Utilities import create_points

import matplotlib.pyplot as plt


pth = 'acoustools/tests/data/svgs/fish.svg'
points, control_points = svg_to_beziers(pth, True)


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]


plt.scatter(xs,ys)
plt.show()