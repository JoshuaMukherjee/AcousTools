if __name__ == '__main__':
    from acoustools.Solvers import naive
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(1,1)
    print(p)
    x = naive(p)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = naive(p)
    print(propagate_abs(x,p))

