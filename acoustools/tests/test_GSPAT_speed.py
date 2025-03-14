from acoustools.Solvers import gspat
from acoustools.Utilities import create_points

import torch
import line_profiler

N=10000

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

run()