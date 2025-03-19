from acoustools.Paths.Bezier import create_bezier_circle, interpolate_bezier
import matplotlib.pyplot as plt

spline = create_bezier_circle(5)


points = []
for bez in spline:
    points += interpolate_bezier(bez,20)

xs = [pt[:,0].item() for pt in points]
ys = [pt[:,1].item() for pt in points]

plt.scatter(xs,ys)
plt.show()