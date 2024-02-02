if __name__ == "__main__":
    from acoustools.Mesh import load_scatterer, get_lines_from_plane,get_centre_of_mass_as_points
    from acoustools.BEM import compute_E, propagate_BEM_pressure
    from acoustools.Utilities import create_points, TOP_BOARD, propagate_abs
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Visualiser import Visualise

    import torch, vedo

    path = "../BEMMedia"
    scatterer = load_scatterer(path+"/Sphere-lam2.stl",dy=-0.06,dz=-0.08)
    # print(get_centre_of_mass_as_points(scatterer))
    # vedo.show(scatterer, axes =1)
    
    N=1
    B=1
    p = create_points(N,B,y=0,x=0,z=0)
    
    E = compute_E(scatterer, p, TOP_BOARD,path=path)
    x = wgs_wrapper(p,board=TOP_BOARD,A=E)
    
    A = torch.tensor((-0.12,0, 0.12))
    B = torch.tensor((0.12,0, 0.12))
    C = torch.tensor((-0.12,0, -0.12))
    normal = (0,1,0)
    origin = (0,0,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path}],vmax=8621, show=True,res=[256,256])


