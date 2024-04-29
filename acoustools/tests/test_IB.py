if __name__ == '__main__':
    from acoustools.Solvers import iterative_backpropagation
    from acoustools.Utilities import create_points, propagate_abs

    p = create_points(2,1)
    print(p)
    x = iterative_backpropagation(p)
    print(propagate_abs(x,p))
    
    p = p.squeeze(0)
    print(p)
    x = iterative_backpropagation(p)
    print(propagate_abs(x,p))

