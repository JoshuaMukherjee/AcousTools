if __name__ == "__main__":

    from acoustools.BEM import  compute_E, propagate_BEM_pressure, BEM_forward_model_grad
    from acoustools.Mesh import get_lines_from_plane, load_multiple_scatterers, scatterer_file_name
    from acoustools.Utilities import create_points, TRANSDUCERS, device, add_lev_sig, forward_model_grad, propagate_abs
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise
    from acoustools.Gorkov import get_finite_diff_points_all_axis
    import acoustools.Constants as Constants

    import vedo, torch
    path = "../BEMMedia"


    wall_paths = [path+"/flat-lam1.stl",path+"/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.175/2,0.175/2],rotys=[90,-90]) #Make mesh at 0,0,0
    walls.scale((1,19/12,19/12),reset=True,origin =False)
    walls.filename = scatterer_file_name(walls)

    N = 1
    B = 1

    # p = create_points(N,B,y=0)
    # p = create_points(N,B,y=0,x=0,z=0.01)
    p = torch.tensor([[0,0],[0,0],[0.04,-0.06]]).unsqueeze(0).to(device)


    E = compute_E(walls, p, board=TRANSDUCERS, path=path, use_cache_H=False)
    x = wgs(p, A=E)
    x = add_lev_sig(x)

    print(torch.abs(E@x))
    print()

    Ex, Ey, Ez = BEM_forward_model_grad(p,walls, TRANSDUCERS, path=path, use_cache_H=False)
    print(torch.abs(Ex@x))
    print(torch.abs(Ey@x))
    print(torch.abs(Ez@x))
    print()

    Fx, Fy, Fz = forward_model_grad(p)
    print(torch.abs(Fx@x))
    print(torch.abs(Fy@x))
    print(torch.abs(Fz@x))
    print()

    
    # # stepsize = 0.000135156253
    # stepsize = Constants.wavelength/8
    # print(stepsize)
    # fin_diff_points = get_finite_diff_points_all_axis(p,stepsize=stepsize)
    # pressures = propagate_BEM_pressure(x,fin_diff_points,walls, path=path,board=TRANSDUCERS)
    # pressure = pressures[:,:N]
    # pressure_FD = pressures[:,N:]
    # split = torch.reshape(pressure_FD,(B,2, -1))
    # grad = (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    # print(torch.abs(grad))
   


    # exit()
    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    # A = torch.tensor((0,-0.09, 0.09))
    # B = torch.tensor((0,0.09, 0.09))
    # C = torch.tensor((0,-0.09, -0.09))
    # normal = (1,0,0)
    # origin = (0,0,0)

    line_params = {"scatterer":walls,"origin":origin,"normal":normal}

    Visualise(A,B,C, x, points=p, colour_functions=[propagate_BEM_pressure, propagate_abs], add_lines_functions=[get_lines_from_plane,None],add_line_args=[line_params,{}],\
              colour_function_args=[{"scatterer":walls,"board":TRANSDUCERS,"path":path},{}],vmax=9000, show=True)
