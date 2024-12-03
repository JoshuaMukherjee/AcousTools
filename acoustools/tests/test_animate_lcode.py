from acoustools.Visualiser import animate_lcode


shape = 'circle'

pth = f'acoustools/tests/data/gcode/{shape}.lcode'
fname = f'acoustools/tests/data/gcode/{shape}.gif'

animate_lcode(pth, skip=1000, fname=fname, extruder=False, xlims=(-0.05, 0.05), ylims=(-0.05, 0.05), zlims=(0, 0.05))

