from acoustools.Paths import svg_to_beziers, OptiSpline, bezier_to_C1, interpolate_bezier
from acoustools.Optimise.OptiSpline_Objectives import optispline_min_distance_control_points


import matplotlib.pyplot as plt

pth = 'acoustools/tests/data/svgs/fish.svg'
points_old, spline_non_C1 = svg_to_beziers(pth, True, dx=-0.06, dy=-0.06)

points_c1, spline =  bezier_to_C1(spline_non_C1)


new_spline = OptiSpline(spline, points_old, optispline_min_distance_control_points,iters=300)
points=[]
for bez in new_spline:
        points += interpolate_bezier(bez, 20)



pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys,marker='.', label='Optimised')


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points_old]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys,marker='.', label='Start')


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points_c1]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys,marker='.', label='C1')

# pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points_opt_c1]
# xs = [p[0] for p in pts]
# ys = [p[1] for p in pts]

# plt.scatter(xs,ys,marker='.', label='C1')

plt.legend()
plt.show()

