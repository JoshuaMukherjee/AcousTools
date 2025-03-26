from acoustools.Utilities import create_points, propagate_abs
from acoustools.Solvers import wgs
from acoustools.Export.Holo import save_holograms, load_holograms
from acoustools.Paths import interpolate_points

import pickle

p1= create_points(1,1,0,0,0.02)
p2= create_points(1,1,0,0,-0.02)


N = 10
ps = interpolate_points(p1,p2, N)
xs = []
for p in ps:
    x = wgs(p)
    xs.append(x)

save_holograms(xs,'output')
pickle.dump(xs, open('output_pickle.pth', 'wb'))