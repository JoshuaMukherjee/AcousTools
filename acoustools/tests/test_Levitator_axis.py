from acoustools.Levitator import LevitatorController
from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs

board = TRANSDUCERS

lev = LevitatorController(ids = (999,1000))

p = create_points(1,1,0,0,0)

x = wgs(p, board = board)
x = add_lev_sig(x)

xs = []

for i in range(100):
    pos = i/10000
    print(pos)
    p = create_points(1,1,0,0,pos)

    x = wgs(p, board = board)
    x = add_lev_sig(x)

    xs.append(x)

print('Ready')

lev.set_frame_rate(200)

lev.levitate(xs[0], sleep_ms=1000)

input()

lev.levitate(xs)

input()

lev.disconnect()



