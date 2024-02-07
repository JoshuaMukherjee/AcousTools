if __name__ == "__main__":
    from acoustools.Gorkov import force_mesh, compute_force
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight
    from acoustools.BEM import compute_E, BEM_forward_model_grad, propagate_BEM_pressure
    import acoustools.Constants as c 
    from acoustools.Visualiser import Visualise

    import vedo, torch

    board = TRANSDUCERS

    path = "../../BEMMedia"
    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths)

    # weight = get_weight(scatterer, c.p_p)
    weight = -1 * (0.1/1000) * 9.81

    p = get_centres_as_points(scatterer)

    E, F,G,H = compute_E(scatterer, p, TRANSDUCERS, return_components=True, path=path)
    x = wgs_wrapper(p, board=board, A=E)

    print(torch.abs(E@x))

    norms = get_normals_as_points(scatterer)

    areas = get_areas(scatterer)
    print(areas.shape)

    force = force_mesh(x,p,norms,areas,board, grad_function=BEM_forward_model_grad, grad_function_args={"scatterer":scatterer,"H":H,"path":path})

    print(force)
    F = torch.sum(force)
    print(F)

    print(F + weight)

