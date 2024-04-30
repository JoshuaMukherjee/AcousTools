if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    import math, pickle

    print('Computing...')
    
    COMPUTE = False
    N = 2000
    radius = 0.02
    
    poss=[]
    xs= []
    if COMPUTE:
        xs = []
        for i in range(N):
            t = ((3.1415926*2) / N) * i
            x = radius * math.sin(t)
            z = radius * math.cos(t)
            poss.append((x,0,z))
            p = create_points(1,1,x=x,y=0,z=z)
            x = wgs(p)
            x = add_lev_sig(x)
            xs.append(x)
            if i % 100 == 0:
                print(i)
        pickle.dump(xs,open('acoustools/tests/data/circle' + str(N) + '.pth','wb'))
    else:
        xs = pickle.load(open('acoustools/tests/data/circle' + str(N) + '.pth','rb'))

    print('Finished Computing \nConnecting to PAT...')

    lev = LevitatorController()
    print('Connected')
    lev.levitate(xs[0])
    
    input()
    print('Moving...')
    try:
        lev.levitate(xs,loop=False)
    except KeyboardInterrupt:
        print('Stopping')
    finally:
        print('Finished Moving')
        input()
        lev.disconnect()



