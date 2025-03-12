from acoustools.Utilities import create_points
from acoustools.Export.lcode import point_to_lcode

p = create_points(2,2)

lcode = point_to_lcode(p)

print(lcode)