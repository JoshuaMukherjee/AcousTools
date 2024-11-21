
from acoustools.Paths import svg_to_beziers, bezier_to_distance, interpolate_bezier
import matplotlib.pyplot as plt


pth = 'acoustools/tests/data/svgs/Complex.svg'
_, bezier_non_c1 = svg_to_beziers(pth, True, dx=-0.06, dy=-0.06)
bezier = bezier_non_c1[0]

points = bezier_to_distance(bezier, max_distance=0.01)

bez_points = interpolate_bezier(*bezier, n=10)

pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in bez_points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys, marker='.')


pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys, marker='.')
plt.show()