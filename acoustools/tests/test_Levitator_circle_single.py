if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, BOTTOM_BOARD
    from acoustools.Solvers import wgs
    import math, pickle, time

    print('Computing...')

    mat_to_world = (-1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1)
    

    board = BOTTOM_BOARD
    
    COMPUTE = False
    N = 200
    radius = 0.02
    
    poss=[]
    xs= []
    if COMPUTE:
        xs = []
        for i in range(N):
            t = ((3.1415926*2) / N) * i
            x = radius * math.sin(t)
            z = radius * math.cos(t)
            poss.append((x,z,0))
            p = create_points(1,1,x=x,y=z,z=0)
            x = wgs(p,board=board)
            xs.append(x)
            if i % 100 == 0:
                print(i)
        pickle.dump(xs,open('acoustools/tests/data/bottom_circle' + str(N) + '.pth','wb'))
    else:
        xs = pickle.load(open('acoustools/tests/data/bottom_circle' + str(N) + '.pth','rb'))
    
    same_32 = []
    for i in range(32):
        same_32.append(xs[0])

    print('Finished Computing \nConnecting to PAT...')
    try:
        lev = LevitatorController(ids=(73,),matBoardToWorld=mat_to_world)
        print('Connected')
        lev.levitate(xs[0])
        input()
        print('Moving...')
        for i in range(100):
            lev.levitate(same_32,num_loops=100)
        # lev.set_frame_rate(10000)
        # lev.levitate(xs,num_loops=100)
    except KeyboardInterrupt:
        print('Stopping')
    except Exception as e:
        print(e)
    finally:
        print('Finished Moving')
        input()
        lev.disconnect()



