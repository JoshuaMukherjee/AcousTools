
if __name__ == "__main__":
    from acoustools.Utilities import forward_model_batched, forward_model_grad, forward_model_second_derivative_unmixed, forward_model_second_derivative_mixed, TRANSDUCERS, create_points, add_lev_sig, propagate
    from acoustools.Solvers import wgs_wrapper
    from acoustools.Gorkov import get_finite_diff_points_all_axis
    from acoustools.Utilities import device, propagate_abs
    import acoustools.Constants as c

    import torch


    def return_mixed_points(points,stepsize):
        '''
        Only works for N=1
        '''
        if points.shape[2] > 1:
            raise RuntimeError("Only for N=1")
        mixed_points = points.clone().repeat((1,1,13))
        #Set x's
        mixed_points[:,0,1] += stepsize
        mixed_points[:,0,2] += stepsize
        mixed_points[:,0,3] -= stepsize
        mixed_points[:,0,4] -= stepsize
        mixed_points[:,0,5] += stepsize
        mixed_points[:,0,6] += stepsize
        mixed_points[:,0,7] -= stepsize
        mixed_points[:,0,8] -= stepsize
        #Set y's
        mixed_points[:,1,1] += stepsize
        mixed_points[:,1,2] -= stepsize
        mixed_points[:,1,3] += stepsize
        mixed_points[:,1,4] -= stepsize
        mixed_points[:,1,9] += stepsize
        mixed_points[:,1,10] += stepsize
        mixed_points[:,1,11] -= stepsize
        mixed_points[:,1,12] -= stepsize
        #Set z's
        mixed_points[:,2,5] += stepsize
        mixed_points[:,2,6] -= stepsize
        mixed_points[:,2,7] += stepsize
        mixed_points[:,2,8] -= stepsize
        mixed_points[:,2,9] += stepsize
        mixed_points[:,2,10] -= stepsize
        mixed_points[:,2,11] += stepsize
        mixed_points[:,2,12] -= stepsize

        return mixed_points


    N=1
    B=1
    D=3
    for i in range(1):
        # points = create_points(N,B,x=0.02, y=-0.005, z=-0.04)
        points = create_points(N,B)
        print(points)
        points = torch.autograd.Variable(points.data, requires_grad=True).to(device).to(torch.complex64)
        x = wgs_wrapper(points).to(torch.complex64)
        activations = add_lev_sig(x)

        board = TRANSDUCERS

        F = forward_model_batched(points,transducers=board)
        Fx, Fy, Fz = forward_model_grad(points,transducers=board)
        Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=board)
        Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points,transducers=board)

        stepsize = 0.000135156253

        fin_diff_points = get_finite_diff_points_all_axis(points,stepsize=stepsize)
        pressure_points = propagate(activations, fin_diff_points)
        pressure = pressure_points[:,:N]
        pressure_fin_diff = pressure_points[:,N:]
        # split = torch.reshape(pressure_fin_diff,(B,2, ((2*D))*N // 2))
        split = torch.reshape(pressure_fin_diff,(B,2, -1))
        grad = (split[:,0,:] - split[:,1,:]) / (2*stepsize)

        p  = torch.abs(F@activations)

        P_a = torch.autograd.grad (p, points, retain_graph=True, create_graph=True)[0]   # creates graph of first derivative
        pa = torch.abs(P_a)

        print("p", p)
        print("Grad","Analytical","Finite Differences","Autograd",sep="\t")
        Px  = torch.abs(Fx@activations)
        print("px", Px, torch.abs(grad[0,0]),pa[:,0],sep="\t")
        Py  = torch.abs(Fy@activations)
        print("py", Py,torch.abs(grad[0,1]),pa[:,1],sep="\t")
        Pz  = torch.abs(Fz@activations)
        print("pz", Pz, torch.abs(grad[0,2]),pa[:,2],sep="\t")
        
        print()

        grad_unmixed = (split[:,0,:] - 2*pressure + split[:,1,:]) / (stepsize**2)

        Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=board)
        Pxx = torch.abs(Fxx@activations)
        print("Pxx", Pxx, torch.abs(grad_unmixed[0,0]), torch.abs(grad_unmixed[0,0]) / Pxx,sep="\t")
        Pyy = torch.abs(Fyy@activations)
        print("Pyy", Pyy, torch.abs(grad_unmixed[0,1]), torch.abs(grad_unmixed[0,1]) / Pyy,sep="\t")
        Pzz = torch.abs(Fzz@activations)
        print("Pzz", Pzz, torch.abs(grad_unmixed[0,2]), torch.abs(grad_unmixed[0,2]) / Pzz,sep="\t")

        print()

        mixed_points = return_mixed_points(points,stepsize)
        mixed_pressure_points = propagate(activations, mixed_points)
              
        Pxy = torch.abs(Fxy@activations)
        mixed_pressure_fin_diff_xy = mixed_pressure_points[:,1:5] * torch.tensor([1,-1,-1,1])
        Pxy_fd = torch.sum(mixed_pressure_fin_diff_xy) / (4*stepsize**2)
        print("Pxy",Pxy, torch.abs(Pxy_fd),torch.abs(Pxy_fd)/Pxy,sep='\t')

        Pxz = torch.abs(Fxz@activations)
        mixed_pressure_fin_diff_xz = mixed_pressure_points[:,5:9] * torch.tensor([1,-1,-1,1])
        Pxz_fd = torch.sum(mixed_pressure_fin_diff_xz) / (4*stepsize**2)
        print("Pxz",Pxz, torch.abs(Pxz_fd), torch.abs(Pxz_fd)/Pxz ,sep='\t')

        Pyz = torch.abs(Fyz@activations)
        mixed_pressure_fin_diff_yz = mixed_pressure_points[:,9:] * torch.tensor([1,-1,-1,1])
        Pyz_fd = torch.sum(mixed_pressure_fin_diff_yz) / (4*stepsize**2)
        print("Pyz",Pyz, torch.abs(Pyz_fd), torch.abs(Pyz_fd)/Pyz ,sep='\t')