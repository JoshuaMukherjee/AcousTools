if __name__ == '__main__':
    from acoustools.Solvers import wgs
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(2,4)
    print(p)
    x = wgs(p)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = wgs(p)
    print(propagate_abs(x,p))

