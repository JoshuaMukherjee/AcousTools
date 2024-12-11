if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig, propagate_abs
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise
    from acoustools.Gorkov import gorkov_analytical

    import torch

    p = create_points(4,1,y=0)
    x = wgs(p)
    x = add_lev_sig(x)

    A = torch.tensor((-0.06,0, 0.06))
    B = torch.tensor((0.06,0, 0.06))
    C = torch.tensor((-0.06,0, -0.06))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=p, colour_functions=[propagate_abs,gorkov_analytical],clr_labels=['Pressure','Gor\'kov'], res=(200,200),link_ax=[0,])