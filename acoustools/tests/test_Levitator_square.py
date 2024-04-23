if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    import time

    xs = []
    pos= [0,0,0]
    step = 0.001
    N = 10

    print('Computing...')

    for i in range(N):
        p = create_points(1,1,x=pos[0],y=pos[1],z=pos[2])
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)

        pos[0] = pos[0] + step
        
    
    for i in range(N):
        p = create_points(1,1,x=pos[0],y=pos[1],z=pos[2])
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)

        pos[2] = pos[2] + step

    for i in range(N):
        p = create_points(1,1,x=pos[0],y=pos[1],z=pos[2])
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)

        pos[0] = pos[0] - step

    for i in range(N):
        p = create_points(1,1,x=pos[0],y=pos[1],z=pos[2])
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)

        pos[2] = pos[2] - step

    print('Finished Computing \nConnecting to PAT...')


    lev = LevitatorController()
    print('Connected')
    lev.levitate(xs[0])
    input()
    print('Moving...')
    for x in xs:
        lev.levitate(x)
        time.sleep(1)
    print('Finished Moving')
    input()
    lev.disconnect()



