from acoustools.Utilities import create_points
from acoustools.Export.lcode import export_to_lcode

N = 10

ps = []
for i in range(N):
    p = create_points(1)
    ps.append(p)

export_to_lcode(f"acoustools/tests/data/lcode/random{N}.lcode", ps,solver='gspat',I=10)