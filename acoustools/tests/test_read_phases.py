if __name__ == "__main__":
    from acoustools.Utilities import read_phases_from_file, TOP_BOARD, create_points,DTYPE
    from acoustools.BEM import load_scatterer, compute_E, propagate_BEM_pressure
    from acoustools.Visualiser import Visualise


    import torch

    path = "../BEMMedia/BEMPhases"
    x = read_phases_from_file(path+"/BunnyCppLam2.csv", top_board=True).to(DTYPE)

    path = "../BEMMedia"
    # scatterer = load_scatterer(path+"/Sphere-lam2.stl",dy=-0.06,dz=-0.08)
    scatterer = load_scatterer(path+"/Bunny-lam2.stl",dz=-0.10, rotz=90)
    # print(get_centre_of_mass_as_points(scatterer))
    # vedo.show(scatterer, axes =1)

    N=1
    B=1
    p = create_points(N,B,y=0,x=0,z=0)

    E, F, G, H = compute_E(scatterer, p, TOP_BOARD,path=path,use_cache_H=False,return_components=True)

    print(torch.abs(E@x))
    # exit()

    A = torch.tensor((-0.12,0, 0.12))
    B = torch.tensor((0.12,0, 0.12))
    C = torch.tensor((-0.12,0, -0.12))
    normal = (0,1,0)
    origin = (0,0,0)

    line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

    # Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure,propagate_abs],colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H},{"board":TOP_BOARD}],vmax=8621, show=True,res=[256,256])
    Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H}],vmax=7000, show=True,res=[256,256])