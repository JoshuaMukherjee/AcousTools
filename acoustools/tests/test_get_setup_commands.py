from acoustools.Export.lcode import get_setup_commands
from acoustools.Utilities import TOP_BOARD, TRANSDUCERS

solver = 'wgs'
I = 100
board = TRANSDUCERS
frame_rate = 1000

commands = get_setup_commands(solver=solver, I=I,board=board, frame_rate=frame_rate )
print(commands)

print('\n\n\n')

solver = 'GORKOV_TARGET'
I = 20
U=-7.5e-5
board = TOP_BOARD
frame_rate = 200
flat_reflector_z = 0

commands = get_setup_commands(solver=solver, I=I, U=U,board=board, frame_rate=frame_rate,flat_reflector_z=flat_reflector_z )
print(commands)