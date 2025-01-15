from acoustools.Levitator import LevitatorController

from acoustools.Utilities import TOP_BOARD, create_points, propagate_abs, add_lev_sig
from acoustools.Solvers import wgs

mat_to_world = (1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)

lev = LevitatorController(ids=(73,), matBoardToWorld=mat_to_world)

board = TOP_BOARD

xs = []
for i in range(200):

    p = create_points(1,1,x=i/1000,y=0,z=0)
    x = wgs(p, board = board)
   # x = add_lev_sig(x,mode='Twin',board=board)
    xs.append(x)
    print(propagate_abs(x, p, board = board))



print('starting')
lev.levitate(xs[0])
input()
lev.levitate(xs,num_loops=100,sleep_ms=1000)
print('done')

input()
print('stopping')

lev.disconnect()

print('Ended')