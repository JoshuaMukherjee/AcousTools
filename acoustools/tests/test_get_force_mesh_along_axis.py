

if __name__ == "__main__":
    from acoustools.Gorkov import force_mesh, compute_force, get_force_mesh_along_axis
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight
    import acoustools.Constants as c 

    import vedo, torch
    import matplotlib.pyplot as plt

    board = TRANSDUCERS

    paths = ["Sphere-lam1.stl"]
    scatterer = load_multiple_scatterers(paths,board, root_path="../BEMMedia/",dys=[-0.06])

    weight = get_weight(scatterer, c.p_p)

    p = get_centres_as_points(scatterer)

    x = wgs_wrapper(p, board=board)

    start = torch.tensor([[-0.06],[0],[0]])
    end = torch.tensor([[0.06],[0],[0]])

    Fxs, Fys, Fzs = get_force_mesh_along_axis(start, end, x, scatterer, board)

    plt.plot(Fzs)
    plt.show()