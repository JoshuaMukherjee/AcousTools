from acoustools.Export.lcode import export_to_lcode
from acoustools.Utilities import TOP_BOARD, TRANSDUCERS
from acoustools.Utilities import create_points

import math 

ps = []

radius = 0.02
Z=0
I = 200

for i in range(I):

        t = ((3.1415926*2) / I) * i
        x = radius * math.sin(t)
        y = radius * math.cos(t)
        p = create_points(1,1,x=x,y=y,z=Z)
        ps.append(p)


export_to_lcode(f"acoustools/tests/data/lcode/circle{I}.lcode",ps, solver='gspat', I=10)