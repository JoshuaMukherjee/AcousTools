from acoustools.Paths import svg_to_beziers, OptiSpline, bezier_to_C1, interpolate_bezier
from acoustools.Optimise.OptiSpline_Objectives import optispline_min_distance_control_points


import matplotlib.pyplot as plt

pth = 'acoustools/tests/data/svgs/fish.svg'
points_old, bezier_non_c1 = svg_to_beziers(pth, True, dx=-0.06, dy=-0.06)

points_c1, bezier =  bezier_to_C1(bezier_non_c1)


new_bezier = OptiSpline(bezier, points_old, optispline_min_distance_control_points,iters=300)
points=[]
for (P0, P3, c11, c12) in new_bezier:
        points += interpolate_bezier(P0,P3, c11, c12, 20)


points_opy_c1, bezier_opt_c1 =  bezier_to_C1(new_bezier)

points_opt_c1 = []
for (P0, P3, c11, c12) in bezier_opt_c1:
        points_opt_c1 += interpolate_bezier(P0,P3, c11, c12, 20)



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

