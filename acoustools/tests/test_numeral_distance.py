from acoustools.Paths import get_numeral, total_distance
from acoustools.Visualiser import ABC

abc = ABC(0.025)

ds= 0

for n in range(1,10):
    print(n, end=' ')
    pth = get_numeral(n, *abc)
    d = total_distance(pth)[0].item()
    print(d)
    ds+=d

print('Average', ds/9)