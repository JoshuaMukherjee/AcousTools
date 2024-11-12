from acoustools.Visualiser import animate_lcode


shape = 'rectangle'

pth = f'acoustools/tests/data/gcode/{shape}.lcode'

animate_lcode(pth, skip=20, fname=f'acoustools/tests/data/gcode/{shape}.gif', extruder=True)

