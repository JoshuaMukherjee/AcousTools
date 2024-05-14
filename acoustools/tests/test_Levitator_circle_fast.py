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
            y = radius * math.cos(t)
            poss.append((x,y,0))
            p = create_points(1,1,x=x,y=y,z=0)
            x = wgs(p)
            x = add_lev_sig(x)
            xs.append(x)
            if i % 100 == 0:
                print(i)
        pickle.dump(xs,open('acoustools/tests/data/circle' + str(N) + '.pth','wb'))
    else:
        xs = pickle.load(open('acoustools/tests/data/circle' + str(N) + '.pth','rb'))

    print('Finished Computing \nConnecting to PAT...')
    try:
        lev = LevitatorController(ids=(73,53))
        print('Connected')
        lev.levitate(xs[0])

        
        input()
        print('Moving...')
   
        print(len(xs))
        lev.set_frame_rate(10)
        lev.levitate(xs,num_loops=1)
        input()
        lev.set_frame_rate(10000)
        lev.levitate(xs,num_loops=10)
    except KeyboardInterrupt:
        print('Stopping')
    except Exception as e:
        print(e)
    finally:
        print('Finished Moving')
        input()
        lev.disconnect()



