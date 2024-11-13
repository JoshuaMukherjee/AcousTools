from acoustools.Visualiser import animate_lcode


shape = 'circle'

pth = f'acoustools/tests/data/gcode/{shape}.lcode'

animate_lcode(pth, skip=20, fname=f'acoustools/tests/data/gcode/{shape}.gif', extruder=False, xlims=(-0.05, 0.05), ylims=(-0.05, 0.05), zlims=(-0.05, 0.05))

