if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, device
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC


    p = create_points(2,1,y=0,z=0)
    x = wgs(p)
    x = add_lev_sig(x)


    Visualise(*ABC(0.1), x, res=(800,800))