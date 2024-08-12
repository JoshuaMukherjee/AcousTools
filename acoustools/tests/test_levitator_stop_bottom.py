from acoustools.Levitator import LevitatorController
mat_to_world = (1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1)

lev = LevitatorController(ids=(73,), matBoardToWorld=mat_to_world)
# lev = LevitatorController(ids=(38,), matBoardToWorld=mat_to_world)
lev.disconnect()