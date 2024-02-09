if __name__ == "__main__":
    from acoustools.Force import force_mesh, compute_force
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight
    from acoustools.BEM import compute_E, BEM_forward_model_grad, propagate_BEM_pressure
    import acoustools.Constants as c 
    from acoustools.Visualiser import Visualise, force_quiver

    import vedo, torch

    board = TRANSDUCERS

    path = "../../BEMMedia"
    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths,dys=[-0.06])

    # weight = get_weight(scatterer, c.p_p)
    weight = -1 * (0.1/1000) * 9.81

    norms = get_normals_as_points(scatterer)
    p = get_centres_as_points(scatterer,add_normals=True)

    E, F,G,H = compute_E(scatterer, p, TRANSDUCERS, return_components=True, path=path)
    x = wgs_wrapper(p, board=board, A=E)

    pres = propagate_BEM_pressure(x,p,scatterer,board=TRANSDUCERS,E=E)

    areas = get_areas(scatterer)
    force = force_mesh(x,p,norms,areas,board, grad_function=BEM_forward_model_grad, F=E, grad_function_args={"scatterer":scatterer,"H":H,"path":path})
    force[force.isnan()] = 0

    F = torch.sum(force)
    print(F)

    print(F + weight)

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


    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TRANSDUCERS,"path":path}],vmax=9000, show=True)

    force_quiver(p,force[:,0,:],force[:,2,:], normal,show=True,log=False)


