from acoustools.Paths import svg_to_beziers, OptiSpline, close_bezier, interpolate_bezier
from acoustools.Optimise.OptiSpline_Objectives import optispline_min_acceleration_position

import matplotlib.pyplot as plt

N = 20

pth = 'acoustools/tests/data/svgs/Complex.svg'
points, bezier = svg_to_beziers(pth, True, dx=-0.06, dy=-0.06,n=N)


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys,marker='.', label='Target')


spline = OptiSpline(bezier, points, optispline_min_acceleration_position,iters=300, objective_params={'alpha':1e-6},n=N, optimise_start=True)
points,spline = close_bezier(spline)


points=[]


for bez in spline:
        points += interpolate_bezier(bez, N)

pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys,marker='.', label='Optimised')



plt.legend()
plt.show()

