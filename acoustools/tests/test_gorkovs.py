

if __name__ == "__main__":
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    from acoustools.Gorkov import gorkov_analytical, gorkov_autograd, gorkov_fin_diff

    N=1
    B=1
    points = create_points(N,B)
    x = wgs(points)
    x = add_lev_sig(x)
    
    U_ag = gorkov_autograd(x,points)
    U_fd = gorkov_fin_diff(x,points)
    U_a = gorkov_analytical(x,points)

    print("Autograd", U_ag.data.squeeze())
    print("Finite Differences",U_fd.data.squeeze())
    print("Analytical",U_a.data.squeeze())