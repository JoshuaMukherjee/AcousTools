from acoustools.Utilities import create_points


xs = [0,1,2,3]
ys = [2,3,4,5]
zs = [2,4,5,6]

ps = create_points(x=xs,y=ys, z=zs)
print(ps)

ps = create_points(1,1)
print(ps)