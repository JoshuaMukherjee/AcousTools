if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    import math

    print('Computing...')

    xs = []
    poss=[]
    N = 250
    radius = 0.02
    for i in range(N):
        t = ((3.1415926*2) / N) * i
        x = radius * math.sin(t)
        z = radius * math.cos(t)
        print(x,0,z)
        poss.append((x,0,z))
        p = create_points(1,1,x=x,y=0,z=z)
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)

    print('Finished Computing \nConnecting to PAT...')


    lev = LevitatorController()
    print('Connected')
    lev.levitate(xs[0])
    
    input()
    print('Moving...')
    try:
        while True:
            lev.levitate(xs)
    except KeyboardInterrupt:
        print('Stopping')
    finally:
        print('Finished Moving')
        input()
        lev.disconnect()



