from acoustools.Levitator import LevitatorController

from acoustools.Utilities import BOTTOM_BOARD, create_points
from acoustools.Solvers import wgs

mat_to_world = (1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1)

lev = LevitatorController(ids=(73,), matBoardToWorld=mat_to_world)

p = create_points(1,1,0,0,0)
x = wgs(p)


lev.levitate(x)

input()

lev.disconnect()