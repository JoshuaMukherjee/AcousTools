if __name__ == "__main__":
    from acoustools.Force import force_mesh, compute_force, force_fin_diff
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight, scale_to_diameter, get_centre_of_mass_as_points,get_lines_from_plane
    from acoustools.BEM import compute_E, BEM_forward_model_grad, propagate_BEM_pressure, BEM_gorkov_analytical
    import acoustools.Constants as c 
    from acoustools.Visualiser import Visualise, force_quiver_3d

    import vedo, torch

    import matplotlib.pyplot as plt

    board = TRANSDUCERS

    path = "../BEMMedia"
    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths,dys=[-0.06])
    scale_to_diameter(scatterer, 0.001)

    # weight = get_weight(scatterer, c.p_p)
    weight = -1 * (0.1/1000) * 9.81

    norms = get_normals_as_points(scatterer)
    p = get_centres_as_points(scatterer)
    com = get_centre_of_mass_as_points(scatterer)

    E, F,G,H = compute_E(scatterer, com, TRANSDUCERS, return_components=True, path=path)
    x = wgs_wrapper(com, board=board, A=E)
    x = add_lev_sig(x)

    pres = propagate_BEM_pressure(x,p,scatterer,board=TRANSDUCERS,E=E)

    areas = get_areas(scatterer)
    # force = force_mesh(x,p,norms,areas,board, grad_function=BEM_forward_model_grad, F=E, grad_function_args={"scatterer":scatterer,"H":H,"path":path})
    force = force_mesh(x,p,norms,areas,board)
    force[force.isnan()] = 0

    F = torch.sum(force,dim=2)
    print(F)

    # print(F + weight)

    size = torch.logspace(-4,4,512)
    print(size[84])
    
    # STEPSIZE = 0.001

    F_A = compute_force(x,com)
    # print(torch.sum(F_A,dim=1))
    print(F_A)

    forces_x = []
    forces_y = []
    forces_z = []
    steps = []

    for s in size:
        F_U_BEM = force_fin_diff(x, com, U_function=BEM_gorkov_analytical,U_fun_args={"scatterer":scatterer, "board":TRANSDUCERS,'path':path},stepsize=s)
        F_U_BEM = torch.abs(F_U_BEM)
        # F_U_BEM = torch.reshape(F_U_BEM, (1,3,-1))
        # print(torch.sum(F_U_BEM,dim=2))
        forces_x.append(F_U_BEM[:,0].cpu().detach().numpy())
        forces_y.append(F_U_BEM[:,1].cpu().detach().numpy())
        forces_z.append(F_U_BEM[:,2].cpu().detach().numpy())
        steps.append(s)
    
    plt.plot(forces_x)
    plt.plot(forces_y)
    plt.plot(forces_z)
    plt.plot(size,[F_A[:,:,0].item()]*len(forces_x))
    plt.plot(size,[F_A[:,:,1].item()]*len(forces_x))
    plt.plot(size,[F_A[:,:,2].item()]*len(forces_x))
    plt.yscale('log')
    # plt.xticks(torch.linspace(0,256,257).cpu().detach().numpy(),size)

    plt.show()

    


