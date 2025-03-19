from acoustools.Solvers import gspat
from acoustools.Utilities import create_points
from acoustools.Levitator import LevitatorController

import torch
import line_profiler

N=10000

lev = LevitatorController(ids=(73,53))

@line_profiler.profile
def run():
    ps = []
    for i in range(N): 
        pnt = create_points(N=3,B=1)
        ps.append(pnt)
        if len(ps) == 32:
            p = torch.concatenate(ps, axis=0)
            ps = []
            x = gspat(p, iterations=10)
            xs = []
            for i in x:
                xs.append(i.unsqueeze(0))
            lev.levitate(xs)

try:
    run()
except KeyboardInterrupt:
    pass