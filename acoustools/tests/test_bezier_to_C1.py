from svgpathtools import svg2paths
from acoustools.Paths import interpolate_bezier, svg_to_beziers, bezier_to_C1
from acoustools.Utilities import create_points

import matplotlib.pyplot as plt


pth = 'acoustools/tests/data/svgs/bez_loop.svg'
points, bezier_non_c1 = svg_to_beziers(pth, True, dx=-0.06, dy=-0.06)
print(bezier_non_c1[0])


points_c1, bezier =  bezier_to_C1(bezier_non_c1)



pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.scatter(xs,ys)

pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points_c1]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.scatter(xs,ys)


plt.show()