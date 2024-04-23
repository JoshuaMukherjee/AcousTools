if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs

    xs = []
    pos= [0,0,0]
    step = 0.001
    N = 50
    
    for i in range(N):
        p = create_points(1,1,x=pos[0],y=pos[1],z=pos[2])
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)

        pos[0] = pos[0] + step



    lev = LevitatorController()
    lev.levitate(xs[0])
    input()
    print('Moving...')
    for x in xs:
        lev.levitate(x)
    print('Finished')
    input()
    lev.disconnect()



