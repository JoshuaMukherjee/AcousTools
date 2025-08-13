if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
    from acoustools.Solvers import wgs

    lev = LevitatorController(ids=(1000,999))

    p = create_points(1,1,x=0,y=0,z=0)
    x = wgs(p)

    print(propagate_abs(x,p))
    print(x.shape)
    x = add_lev_sig(x)

    lev.levitate(x)
    print('Levitating...')
    input()
    print('Stopping...')
    lev.disconnect()
    print('Stopped')



