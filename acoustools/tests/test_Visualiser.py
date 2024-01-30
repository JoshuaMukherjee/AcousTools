if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Visualiser import Visualise

    import torch

    p = create_points(1,1,x=0,y=0,z=0)
    x = wgs_wrapper(p)
    x = add_lev_sig(x)

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=p)