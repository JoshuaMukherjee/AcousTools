if __name__ == "__main__":
    from acoustools.Mesh import load_scatterer, get_lines_from_plane,get_centre_of_mass_as_points, scale_to_diameter
    from acoustools.BEM import compute_E, propagate_BEM_pressure, compute_H
    from acoustools.Utilities import create_points, TOP_BOARD, propagate_abs
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise, ABC
    import acoustools.Constants as c

    from torch.profiler import profile, record_function, ProfilerActivity

    import torch, vedo



    # @profile
    def run():
        path = "../BEMMedia"
        scatterer = load_scatterer(path+"/Sphere-lam2.stl",dy=-0.06,dz=-0.08)
        # scale_to_diameter(scatterer, 2*c.R)
        # scatterer = load_scatterer(path+"/Bunny-lam2.stl",dz=-0.10, rotz=90)
        # print(get_centre_of_mass_as_points(scatterer))
        # vedo.show(scatterer, axes =1)
        
        N=1
        B=1
        p = create_points(N,B,y=0,x=0,z=0)
        # p = create_points(N,B,y=0) 
        
        # E = compute_E(scatterer, p, TOP_BOARD,path=path,use_cache_H=False)
        H = compute_H(scatterer, TOP_BOARD)
        E, F, G, H = compute_E(scatterer, p, TOP_BOARD,path=path,use_cache_H=False,return_components=True,H=H)
        x = wgs(p,board=TOP_BOARD,A=E)
        # print(x.shape)
        
        A,B,C = ABC(0.12)
        normal = (0,1,0)
        origin = (0,0,0)

        line_params = {"scatterer":scatterer,"origin":origin,"normal":normal}

        # Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure,propagate_abs],
        # colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H},
        # {"board":TOP_BOARD}],vmax=8621, show=True,res=[256,256])
        
        Visualise(A,B,C, x, colour_functions=[propagate_BEM_pressure],
                  colour_function_args=[{"scatterer":scatterer,"board":TOP_BOARD,"path":path,'H':H}],
                  vmax=8621,res=[256,256], show=True)

        # d = torch.device('cuda:0')
        # free, total = torch.cuda.mem_get_info(d)
        # mem_used_MB = (total - free) / 1024 ** 2
        # print(mem_used_MB)


    run()



