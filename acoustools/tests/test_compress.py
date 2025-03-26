from acoustools.Utilities import create_points, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Export.Holo import save_holograms, load_holograms

import torch

p = create_points(1,1)
x = wgs(p)
print(x.shape)

save_holograms(x,'output')
xs = load_holograms('output')

print(propagate_abs(x,p))
print(propagate_abs(xs[0],p))