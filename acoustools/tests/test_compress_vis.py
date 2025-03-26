from acoustools.Utilities import create_points, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Export.Holo import save_holograms, load_holograms
from acoustools.Visualiser import ABC, Visualise

import torch

p = create_points(1,1, y=0)
x = wgs(p)
print(x.shape)

save_holograms(x,'output')
xs = load_holograms('output')

print(propagate_abs(x,p))
print(propagate_abs(xs[0],p))

abc = ABC(0.1)

Visualise(*abc,x)
Visualise(*abc,xs[0])