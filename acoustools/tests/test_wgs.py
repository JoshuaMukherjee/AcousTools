from acoustools.Solvers import wgs_wrapper
from acoustools.Utilities import create_points, propagate_abs, write_to_file

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    p = create_points(1,1,0,0,0)

    x = wgs_wrapper(p)

    write_to_file(x, 'wgsphases.csv',1)
