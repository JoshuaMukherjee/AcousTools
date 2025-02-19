if __name__ == "__main__":

    from acoustools.BEM import load_scatterer, scatterer_file_name, compute_E, propagate_BEM_pressure, BEM_forward_model_grad
    from acoustools.Mesh import get_lines_from_plane
    from acoustools.Utilities import create_points, TRANSDUCERS, device, add_lev_sig, forward_model_grad, propagate_abs, TOP_BOARD, BOTTOM_BOARD
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    from acoustools.Gorkov import get_finite_diff_points_all_axis
    import acoustools.Constants as Constants

    import vedo, torch
    path = "../BEMMedia"

    USE_CACHE = True
    board = BOTTOM_BOARD

    sphere_pth =  path+"/Sphere-lam2.stl"
    sphere = load_scatterer(sphere_pth, dy=-0.06, dz=0.0) #Make mesh at 0,0,0

    # vedo.show(sphere, axes=1)
    # exit()

    N = 1
    B = 1

    # p = create_points(N,B,y=0)
    p = create_points(N,B,y=0,x=0,z=-0.04)
    # p = torch.tensor([[0,0],[0,0],[-0.06]]).unsqueeze(0).to(device)


    E = compute_E(sphere, p, board=board, path=path, use_cache_H=USE_CACHE)
    x = wgs(p, A=E)

    print(torch.abs(E@x))
    print()

    Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H = BEM_forward_model_grad(p,sphere, board, path=path, use_cache_H=USE_CACHE, return_components=True)
    print(torch.abs(Ex@x))
    print(torch.abs(Ey@x))
    print(torch.abs(Ez@x))
    print()

    print(torch.abs(Gx@H@x))
    print(torch.abs(Gy@H@x))
    print(torch.abs(Gz@H@x))
    print()

    print(torch.abs(Fx@x))
    print(torch.abs(Fy@x))
    print(torch.abs(Fz@x))
    print()

    PMx, PMy, PMz = forward_model_grad(p, transducers=board)
    print(torch.abs(PMx@x))
    print(torch.abs(PMy@x))
    print(torch.abs(PMz@x))
    print()


    # exit()

    abc = ABC(0.07)
    
    normal = (0,1,0)
    origin = (0,0,0)


    line_params = {"scatterer":sphere,"origin":origin,"normal":normal}

    Visualise(*abc, x, points=p, colour_functions=[propagate_BEM_pressure, propagate_abs],colour_function_args=[{"scatterer":sphere,"board":board,"path":path},{"board":board}],vmax=9000, show=True)
