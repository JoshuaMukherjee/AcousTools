if __name__ == '__main__':
    from acoustools.Levitator import LevitatorController
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs_wrapper

    lev = LevitatorController()

    p = create_points(1,1,x=0,y=0,z=0)
    x = wgs_wrapper(p)
    x = add_lev_sig(x)

    lev.levitate(x)
    input()
    lev.disconnect()



