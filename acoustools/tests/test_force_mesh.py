if __name__ == "__main__":
    from acoustools.Force import force_mesh, compute_force, force_fin_diff
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight, scale_to_diameter, get_centre_of_mass_as_points,get_lines_from_plane
    from acoustools.BEM import compute_E, BEM_forward_model_grad, propagate_BEM_pressure, BEM_gorkov_analytical, get_cache_or_compute_H_gradients
    import acoustools.Constants as c 
    from acoustools.Visualiser import Visualise, force_quiver_3d, force_quiver

    import vedo, torch


    board = TRANSDUCERS

    path = "../BEMMedia"
    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths,dys=[-0.06])
    # scale_to_diameter(scatterer, 0.001)
    scale_to_diameter(scatterer, 2*c.R)
    print(get_centre_of_mass_as_points(scatterer))
    # scale_to_diameter(scatterer, 6*c.R)

    # vedo.write(scatterer,path+"/TinySphere-lam2.stl",False)
    # exit()

    weight = -1*get_weight(scatterer, c.p_p)
    print(weight)
    # weight = -1 * (0.1/1000) * 9.81

    norms = get_normals_as_points(scatterer)
    p = get_centres_as_points(scatterer)
    com = get_centre_of_mass_as_points(scatterer)

    E, F,G,H = compute_E(scatterer, com, TRANSDUCERS, return_components=True, path=path, print_lines=False)
    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=False,path=path)
    x = wgs(com, board=board, A=E)
    x = add_lev_sig(x)
    pres = propagate_BEM_pressure(x,p,scatterer,board=TRANSDUCERS,E=E)

    areas = get_areas(scatterer)
    # force = force_mesh(x,p,norms,areas,board, grad_function=BEM_forward_model_grad, F=E, grad_function_args={"scatterer":scatterer,"H":H,"path":path, F=H})
    # force = force_mesh(x,p,norms,areas,board)
    force = force_mesh(x,p,norms,areas,board,None,None,Ax=Hx, Ay=Hy, Az=Hz,F=H)
    force[force.isnan()] = 0

    F = torch.sum(force,dim=2)
    print(F)

    # print(F + weight)

    # STEPSIZE = 0.000135156253*64
    # STEPSIZE = c.wavelength/5
    STEPSIZE = 0.0021
    
    F_U_BEM = force_fin_diff(x, com, U_function=BEM_gorkov_analytical,U_fun_args={"scatterer":scatterer, "board":TRANSDUCERS,'path':path},stepsize=STEPSIZE)
    # F_U_BEM = torch.reshape(F_U_BEM, (1,3,-1))
    # print(torch.sum(F_U_BEM,dim=2))
    print(F_U_BEM)

    F_U_PM_FD = force_fin_diff(x, com,stepsize=STEPSIZE)
    # F_U_PM_FD = torch.reshape(F_U_PM_FD, (1,3,-1))
    # print(torch.sum(F_U_PM_FD,dim=2))
    print(F_U_PM_FD)

    F_A = compute_force(x,com)
    # print(torch.sum(F_A,dim=1))
    print(F_A)

    # A = torch.tensor((-0.09,0, 0.09))
    # B = torch.tensor((0.09,0, 0.09))
    # C = torch.tensor((-0.09,0, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)

    A = torch.tensor((-0.01,0, 0.01))
    B = torch.tensor((0.01,0, 0.01))
    C = torch.tensor((-0.01,0, -0.01))
    normal = (0,1,0)
    origin = (0,0,0)

    # A = torch.tensor((0,-0.09, 0.09))
    # B = torch.tensor((0,0.09, 0.09))
    # C = torch.tensor((0,-0.09, -0.09))
    normal = (1,0,0)
    # origin = (0,0,0)


    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}
    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TRANSDUCERS,"path":path}],vmax=9000, show=True,add_lines_functions=[get_lines_from_plane], add_line_args=[line_params])

    # force_quiver(p,force[:,0,:],force[:,2,:], normal)


