if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
    from acoustools.Solvers import wgs

    import math, pickle, time


    print('Computing...')
    xs = []
    poss=[]
    N = 1000
    radius = 0.02
    COMPUTE = True
    if COMPUTE:
        for i in range(N):
            t = ((3.1415926*2) / N) * i
            x = radius * math.sin(t)
            z = radius * math.cos(t)
            print(i,x,0,z, end='                \r')
            poss.append((x,0,z))
            p = create_points(1,1,x=x,y=0,z=z)
            x = wgs(p)
            x = add_lev_sig(x, mode='Eye')
            xs.append(x)
            pickle.dump(xs,open('acoustools/tests/data/circleVertEye' + str(N) + '.pth','wb'))
    else:
        xs = pickle.load(open('acoustools/tests/data/circleVertEye' + str(N) + '.pth','rb'))
    print('Finished\nConnecting...')


    try:
        lev = LevitatorController(ids=(73,53))
        print('Connected')
        lev.levitate(xs[0])

        
        input()
        print('Moving...')
   
        print(len(xs))
        # lev.set_frame_rate(10)
        # lev.levitate(xs,num_loops=1)

        start = time.time_ns()
        lev.set_frame_rate(2000)
        lev.levitate(xs,num_loops=10)
        end = time.time_ns()
        print((end-start)/1e9, 'Seconds')
    except KeyboardInterrupt:
        print('Stopping')
    except Exception as e:
        print(e)
    finally:
        print('Finished Moving')
        input()
        lev.disconnect()



