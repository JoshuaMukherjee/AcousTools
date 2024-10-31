if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    import time, pickle

    xs = []
    pos= [0,0,0]
    edge = 0.02
    N = 400
    step = edge / N

    compute = False
    if compute:
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
        
        pickle.dump(xs,open('acoustools/tests/data/square' + str(N) + '.pth','wb'))
    else:
        xs = pickle.load(open('acoustools/tests/data/square' + str(N) + '.pth','rb'))
    
    print(len(xs))

    print('Finished Computing \nConnecting to PAT...')


    lev = LevitatorController(ids=(73,53))
    lev.set_frame_rate(4000)

    print('Connected')
    lev.levitate(xs[0])
    input()
    print('Moving...')

  
    lev.levitate(xs, num_loops=30)
        # time.sleep(0.2)
    print('Finished Moving')
    input()
    lev.disconnect()



