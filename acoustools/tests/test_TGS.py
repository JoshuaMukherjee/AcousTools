
if __name__ == '__main__':

    from acoustools.Utilities import create_points,forward_model_batched,TRANSDUCERS, device, add_lev_sig
    from acoustools.Solvers import naive_solver_batched, wgs_solver_batch, temporal_wgs
    from acoustools.Visualiser import Visualise

    import torch
    

    N=4
    p = create_points(N,1,y=0)
    F = forward_model_batched(p,TRANSDUCERS)

    _,_,x_wgs = wgs_solver_batch(F, torch.ones(N,1).to(device)+0j,200)
    x_wgs = add_lev_sig(x_wgs)


    T_in = torch.pi/64 #Hologram phase change threshold
    T_out = 0 #Point activations phase change threshold

    p = p + 0.0005 #Move particles a small amount - 0.5mm
    F = forward_model_batched(p,TRANSDUCERS)
    _,_,x = temporal_wgs(F,torch.ones(N,1).to(device)+0j,200, x_wgs, F@x_wgs, T_in, T_out)
    # x = add_lev_sig(x)

    A = torch.tensor((-0.09, 0, 0.09), device=device)
    B = torch.tensor((0.09, 0, 0.09), device=device)
    C = torch.tensor((-0.09, 0, -0.09), device=device)
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=p,vmax=5000)