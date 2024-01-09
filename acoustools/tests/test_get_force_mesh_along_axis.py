

if __name__ == "__main__":
    from acoustools.Gorkov import force_mesh, compute_force, get_force_mesh_along_axis
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, TRANSDUCERS
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Mesh import load_multiple_scatterers, get_normals_as_points, get_centres_as_points, get_areas, get_weight, load_scatterer, scale_to_diameter, merge_scatterers
    import acoustools.Constants as c 

    import vedo, torch
    import matplotlib.pyplot as plt

    board = TRANSDUCERS

    wall_paths = ["../BEMMedia/flat-lam1.stl","../BEMMedia/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.06,0.06],rotys=[90,-90]) #Make mesh at 0,0,0
    
    
    ball_path = "../BEMMedia/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.04)

    scatterer = merge_scatterers(ball, walls)

    weight = get_weight(ball, c.p_p)

    p = get_centres_as_points(scatterer)

    x = wgs_wrapper(p, board=board)

    start = torch.tensor([[-0.04],[0],[0]])
    end = torch.tensor([[0.04],[0],[0]])

    Fxs, Fys, Fzs = get_force_mesh_along_axis(start, end, x, [ball,walls], board, path = "../BEMMedia/")
    Fxs = [f.cpu().detach().numpy() for f in Fxs]
    Fys = [f.cpu().detach().numpy() for f in Fys]
    Fzs = [f.cpu().detach().numpy() - weight for f in Fzs]


    plt.subplot(3,1,1)
    plt.plot(Fxs)
    plt.subplot(3,1,2)
    plt.plot(Fys)
    plt.subplot(3,1,3)
    plt.plot(Fzs)
    
    plt.show()