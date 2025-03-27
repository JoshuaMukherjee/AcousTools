if __name__ == "__main__":

    from acoustools.BEM import  compute_E, propagate_BEM_pressure, BEM_forward_model_grad
    from acoustools.Mesh import get_lines_from_plane, load_scatterer, scatterer_file_name
    from acoustools.Utilities import create_points, TRANSDUCERS, device, add_lev_sig, forward_model_grad, propagate_abs, TOP_BOARD, BOTTOM_BOARD
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    from acoustools.Gorkov import get_finite_diff_points_all_axis, get_finite_diff_points_all_axis
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
    # p = create_points(N,B,y=0,x=0,z=-0.04)
    p = create_points(N,B)
    # p = torch.tensor([[0,0],[0,0],[-0.06]]).unsqueeze(0).to(device)


    E,F,G,H = compute_E(sphere, p, board=board, path=path, use_cache_H=USE_CACHE, return_components=True)
    x = wgs(p, A=E)

    print(torch.abs(E@x))
    print()

    Ex, Ey, Ez, Fx, Fy, Fz, Gx, Gy, Gz, H = BEM_forward_model_grad(p,sphere, board, path=path, use_cache_H=USE_CACHE, return_components=True)

    PGx = torch.abs(Gx@H@x).squeeze()
    PGy = torch.abs(Gy@H@x).squeeze()
    PGz = torch.abs(Gz@H@x).squeeze()

    PEx = torch.abs(Ex@x).squeeze()
    PEy = torch.abs(Ey@x).squeeze()
    PEz = torch.abs(Ez@x).squeeze()

   

    PFx = torch.abs(Fx@x).squeeze()
    PFy = torch.abs(Fy@x).squeeze()
    PFz = torch.abs(Fz@x).squeeze()

    # f_grad = torch.stack((Fx@x, Fy@x, Fz@x)).reshape((3,1))

    # PMx, PMy, PMz = forward_model_grad(p, transducers=board)
    # print(torch.abs(PMx@x))
    # print(torch.abs(PMy@x))
    # print(torch.abs(PMz@x))
    # print()

    step = 0.000135156253 
    ps = get_finite_diff_points_all_axis(p, stepsize=step)
    Efd,Ffd,Gfd,Hfd = compute_E(sphere, ps, board=board, path=path, use_cache_H=USE_CACHE, return_components=True)
    
    x_fd = wgs(p, A=E)
    
    Fx = Ffd@x_fd
    p = Fx[:,0:N,:]
    Fx_fd = Fx[:,N:,:].reshape(2,3,N)

    Fx_fd_1 = Fx_fd[0,:,:]
    Fx_fd_2 = Fx_fd[1,:,:]
    
    f_fd_grad = (Fx_fd_1-Fx_fd_2)/(2*step)
    print("F FD \t Analytical \t Ratio")
    f_fd = torch.abs(f_fd_grad)
    print(f_fd[0], PFx, f_fd[0]/PFx)
    print(f_fd[1], PFy, f_fd[1]/PFy)
    print(f_fd[2], PFz, f_fd[2]/PFz)
    print()


    GH = Gfd@Hfd

    GHx = GH@x_fd
    p = GHx[:,0:N,:]
    GHx_fd = GHx[:,N:,:].reshape(2,3,N)

    GHx_fd_1 = GHx_fd[0,:,:]
    GHx_fd_2 = GHx_fd[1,:,:]
    
    gh_grad = (GHx_fd_1-GHx_fd_2)/(2*step)
    print("GH FD  \t Analytical \t Ratio")
    gh_grad_abs = torch.abs(gh_grad)
    print(gh_grad_abs[0], PGx, gh_grad_abs[0]/PGx)
    print(gh_grad_abs[1], PGy, gh_grad_abs[1]/PGy)
    print(gh_grad_abs[2], PGz, gh_grad_abs[2]/PGz)
    print()


    Efdx = Efd@x_fd
    pE = Efdx[:,0:N,:]
    Efd_fd = Efdx[:,N:,:].reshape(2,3,N)

    Efd_fd_1 = Efd_fd[0,:,:]
    Efd_fd_2 = Efd_fd[1,:,:]
    
    e_grad = (Efd_fd_1-Efd_fd_2)/(2*step)
    print('E FD')
    e_grad_abs = torch.abs(e_grad)
    print(e_grad_abs[0], PEx, e_grad_abs[0]/PEx)
    print(e_grad_abs[1], PEy, e_grad_abs[1]/PEy)
    print(e_grad_abs[2], PEz, e_grad_abs[2]/PEz)
    print()

    print()

    print("F + GH fd")
    print(torch.abs(f_fd_grad + gh_grad))

    # print(torch.abs(f_grad + gh_grad) / torch.abs(e_grad))
    
    exit()
   


    def propagate_GH(activations, points):
        E,F,G,H = compute_E(sphere, points, board=board, path=path, use_cache_H=USE_CACHE, return_components=True)
        
        return torch.abs(G@H@activations)


    # exit()

    abc = ABC(0.07)
    normal = (0,1,0)
    origin = (0,0,0)


    line_params = {"scatterer":sphere,"origin":origin,"normal":normal}

    Visualise(*abc, x, colour_functions=[propagate_BEM_pressure, propagate_abs,propagate_GH],colour_function_args=[{"scatterer":sphere,"board":board,"path":path},{"board":board},{}],vmax=9000, show=True)
