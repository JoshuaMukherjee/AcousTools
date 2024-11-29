from acoustools.Fabrication.Optimise import interweave_points

shape = 'circle'

pth = f'acoustools/tests/data/gcode/{shape}.lcode'

interweave_points(pth,num_points=2)