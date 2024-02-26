if __name__ == "__main__":
    from acoustools.Utilities import add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_weight, scale_to_diameter, get_centre_of_mass_as_points,get_lines_from_plane
    from acoustools.BEM import compute_E, propagate_BEM_pressure
    import acoustools.Constants as c 
    from acoustools.Visualiser import Visualise

    import vedo, torch


    board = TRANSDUCERS

    path = "../BEMMedia"
    paths = [path+"/Sphere-lam2.stl"]
    scatterer = load_multiple_scatterers(paths,dys=[-0.06])
    scale_to_diameter(scatterer, 2*c.R)
    print(get_centre_of_mass_as_points(scatterer))


    weight = -1*get_weight(scatterer, c.p_p)
    print(weight)
    # weight = -1 * (0.1/1000) * 9.81

    norms = get_normals_as_points(scatterer)
    p = get_centres_as_points(scatterer)
    com = get_centre_of_mass_as_points(scatterer)

    E, F,G,H = compute_E(scatterer, com, TRANSDUCERS, return_components=True, path=path, print_lines=True)

    x = wgs_wrapper(com, board=board, A=E)
    x = add_lev_sig(x)

    print(propagate_BEM_pressure(x, com, scatterer, board=TRANSDUCERS, use_cache_H=True, path=path, print_lines=True))
    print()
    print(propagate_BEM_pressure(x, com, scatterer, board=TRANSDUCERS, use_cache_H=False, path=path, print_lines=True))


    A = torch.tensor((-0.01,0, 0.01))
    B = torch.tensor((0.01,0, 0.01))
    C = torch.tensor((-0.01,0, -0.01))
    normal = (0,1,0)
    origin = (0,0,0)


    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}
    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure, propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TRANSDUCERS,"path":path}, {"scatterer":scatterer,"board":TRANSDUCERS,"path":path,"use_cache_H":False}],vmax=9000, show=True,add_lines_functions=[get_lines_from_plane, get_lines_from_plane], add_line_args=[line_params,line_params])
