from acoustools.Intepreter import read_lcode 
from acoustools.Export import save_holograms, load_holograms
import time

import matplotlib.pyplot as plt


I = 100
path = f"acoustools/tests/data/lcode/random{I}.lcode"

times= []

for i in range(32):
    start = time.time_ns()
    xs = read_lcode(path, ids=(-1,), points_per_batch=i)
    end = time.time_ns()

    times.append((end-start)/(I*1e9))

plt.scatter(range(32),times)
plt.show()