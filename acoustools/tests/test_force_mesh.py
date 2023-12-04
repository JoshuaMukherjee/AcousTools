if __name__ == "__main__":
    from acoustools.Gorkov import force_mesh, compute_force
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.BEM import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight
    import acoustools.Constants as c 

    import vedo, torch

    board = TRANSDUCERS

    paths = ["Sphere-lam1.stl"]
    scatterer = load_multiple_scatterers(paths,board, root_path="/Users/joshuamukherjee/Desktop/Education/University/UCL/PhD/BEMMedia/")

    weight = get_weight(scatterer, c.p_p)
    print(weight)

    p = get_centres_as_points(scatterer)

    x = wgs_wrapper(p, board=board)

    norms = get_normals_as_points(scatterer)

    areas = get_areas(scatterer)
    print(areas.shape)

    force = force_mesh(x,p,norms,areas,board)

    print(force)
    F = torch.sum(force)
    print(F)

    print(F - weight)

