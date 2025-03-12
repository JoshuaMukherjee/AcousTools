from acoustools.Export.lcode import export_to_lcode
from acoustools.Utilities import TOP_BOARD, TRANSDUCERS
from acoustools.Utilities import create_points

p = create_points(2,2)

solver = 'wgs'
I = 100
board = TRANSDUCERS
frame_rate = 1000

export_to_lcode("acoustools/tests/data/lcode/test_export.lcode",p,solver=solver,I=I, board=board,frame_rate=frame_rate)