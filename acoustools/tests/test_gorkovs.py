

if __name__ == "__main__":
    from acoustools.Utilities import create_points, forward_model, add_lev_sig
    from acoustools.Solvers import wgs_wrapper, wgs
    import matplotlib.pyplot as plt
    from acoustools.Gorkov import gorkov_analytical, gorkov_autograd, gorkov_fin_diff

    N=1
    B=1
    F_As = []
    F_FDs = []
    F_aFDs = []
    axis=0

    points = create_points(N,B)
    x = wgs_wrapper(points)
    x = add_lev_sig(x)
    
    U_ag = gorkov_autograd(x,points)
    U_fd = gorkov_fin_diff(x,points)
    U_a = gorkov_analytical(x,points)

    print("Autograd", U_ag.data.squeeze())
    print("Finite Differences",U_fd.data.squeeze())
    print("Analytical",U_a.data.squeeze())