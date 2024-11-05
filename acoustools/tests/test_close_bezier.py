from acoustools.Paths import svg_to_beziers, close_bezier

import matplotlib.pyplot as plt

pth = 'acoustools/tests/data/svgs/complex.svg'
points, bezier = svg_to_beziers(pth, True, dx=-0.06, dy=-0.06)
print(bezier)

points,bezier = close_bezier(bezier)
print(bezier)

pts = [[p.detach().cpu()[:,0].item(),p.detach().cpu()[:,1].item()] for p in points]
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]

plt.plot(xs,ys,marker='.', label='Target')





plt.legend()
plt.show()

