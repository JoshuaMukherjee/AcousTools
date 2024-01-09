if __name__ == "__main__":
    from acoustools.Gorkov import force_mesh, compute_force, torque_mesh
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS, device
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight, get_centre_of_mass_as_points
    import acoustools.Constants as c 

    import vedo, torch

    board = TRANSDUCERS

    paths = ["Sphere-lam1.stl"]
    scatterer = load_multiple_scatterers(paths, root_path="../BEMMedia/")

    p = get_centres_as_points(scatterer)

    x = wgs_wrapper(p, board=board)

    norms = get_normals_as_points(scatterer)

    areas = get_areas(scatterer)
    print(areas.shape)

    centre_of_mass = get_centre_of_mass_as_points(scatterer)
    torque = torque_mesh(x,p,norms,areas,centre_of_mass,board)

    print(torque)
    F = torch.sum(torque)
    print(F)


