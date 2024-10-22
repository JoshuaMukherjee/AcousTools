if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
    from acoustools.Solvers import wgs

    import time

    lev = LevitatorController(ids=(73,53))

    xs = []

    p = create_points(1,1,x=0,y=0,z=0)
    start = wgs(p)
    print(propagate_abs(start,p))
    start = add_lev_sig(start)

    for i in range(20):
        p = create_points(1,1,x=0,y=0,z=0.0005*i)
        x = wgs(p)
        x = add_lev_sig(x)
        print(propagate_abs(x,p))
        xs.append(x)

    lev.levitate(start)
    print('Levitating...')
    input()
    print('Moving...')
    for x in xs:
        lev.levitate(x)
        time.sleep(0.1)
    input()
    print('Stopping...')
    lev.disconnect()
    print('Stopped')



