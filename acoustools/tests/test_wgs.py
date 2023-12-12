from acoustools.Solvers import wgs_wrapper
from acoustools.Utilities import create_points, propagate_abs

if __name__ == "__main__":
    p = create_points(4,2)
    print(propagate_abs(wgs_wrapper(p),p))