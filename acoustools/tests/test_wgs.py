from acoustools.Solvers import wgs_wrapper
from acoustools.Utilities import create_points, propagate_abs

import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    means = []
    for n in range(4,100):
        p = create_points(n,1)
        m = torch.mean(propagate_abs(wgs_wrapper(p),p))
        means.append(m)
        print(n)
    
    plt.plot([m.cpu().detach().numpy() for m in means])
    plt.ylabel("Mean Pressure (Pa)")
    plt.xlabel("No. Points")

    plt.yscale("log")
    plt.xscale("log")
    plt.show()

