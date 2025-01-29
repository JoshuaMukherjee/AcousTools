if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig, write_to_file
    from acoustools.Solvers import wgs
    import math, time

    print('Computing...')
    xs = []
    poss=[]
    N = 250
    radius = 0.02
    for i in range(N):
        t = ((3.1415926*2) / N) * i
        x = radius * math.sin(t)
        z = radius * math.cos(t)
        print(i,x,0,z)
        poss.append((x,0,z))
        p = create_points(1,1,x=x,y=0,z=z)
        x = wgs(p)
        x = add_lev_sig(x)
        xs.append(x)
    print('Finished\nConnecting...')

    lev = LevitatorController(ids=(999,1000))

    lev.levitate(xs[0])
    print(poss[0])
    input()
    print('Moving...')
    try:
        while True:
            for x in xs:
                lev.levitate(x)
                time.sleep(0.1)
    except:  
        print('Finished')
        input()
        lev.disconnect()



