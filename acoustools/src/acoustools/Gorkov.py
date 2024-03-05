import torch
from acoustools.Utilities import device, propagate, propagate_abs, add_lev_sig, forward_model_batched, forward_model_grad, TRANSDUCERS, forward_model_second_derivative_unmixed,forward_model_second_derivative_mixed, return_matrix, DTYPE
import acoustools.Constants as c
from acoustools.BEM import grad_2_H, grad_H, get_cache_or_compute_H, get_cache_or_compute_H_gradients
from acoustools.Mesh import translate, get_centre_of_mass_as_points, get_centres_as_points, get_normals_as_points, get_areas, merge_scatterers

def gorkov_autograd(activation, points, K1=None, K2=None, retain_graph=False,board=TRANSDUCERS,**params):
    '''
    Computes the Gorkov potential using pytorch's autograd system\\
    `activation` The transducer activations to use \\
    `points` The points to compute the potential at\\
    `K1` The value for K1 in the Gorkov equation, if None will use `c.V / (4*c.p_0*c.c_0**2)`
    `K2` The value for K2 in the Gorkov equation, if None will use `3*c.V / (4*(2*c.f**2 * c.p_0))`\\
    `board` The transducer boards to use\\
    `retain_graph` Value will be passed to autograd\\
    Returns gorkov potential at each point
    '''

    var_points = torch.autograd.Variable(points.data, requires_grad=True).to(device).to(DTYPE)

    B = points.shape[0]
    N = points.shape[2]
    
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)    
    
    pressure = propagate(activation.to(DTYPE),var_points,board=board)
    pressure.backward(torch.ones((B,N))+0j, inputs=var_points, retain_graph=retain_graph)
    grad_pos = var_points.grad

    if K1 is None:
        # K1 = 1/4 * c.V * (1/(c.c_0**2 * c.p_0) - 1/(c.c_p**2 * c.p_p))
        K1 = c.V / (4*c.p_0*c.c_0**2) #Assuming f1=f2=1
    if K2 is None:
        # K2 = 3/4 * c.V * ((c.p_0-c.p_p) / (c.f**2 * c.p_0 * (c.p_0+2*c.p_p)) )
        K2 = 3*c.V / (4*(2*c.f**2 * c.p_0)) #Assuming f1=f2=1


    gorkov = K1 * torch.abs(pressure) **2 - K2 * torch.sum((torch.abs(grad_pos)**2),1)
    return gorkov

def get_finite_diff_points(points, axis, stepsize = 0.000135156253):
    '''
    Gets points for finite difference calculations in one axis\\
    `points` Points around which to find surrounding points\\
    `axis` The axis to add points in\\
    `stepsize` The distance aroud points to add, default 0.000135156253\\
    Returns points 
    '''
    #points = Bx3x4
    points_h = points.clone()
    points_neg_h = points.clone()
    points_h[:,axis,:] = points_h[:,axis,:] + stepsize
    points_neg_h[:,axis,:] = points_neg_h[:,axis,:] - stepsize

    return points_h, points_neg_h

def get_finite_diff_points_all_axis(points,axis="XYZ", stepsize = 0.000135156253):
    '''
    Gets points for finite difference calculations\\
    `points` Points around which to find surrounding points\\
    `axis` The axes to add points in as a string containing 'X', 'Y' and/or 'Z' eg 'XYZ' will use all three axis but 'YZ' will only add points in the YZ axis\\
    `stepsize` The distance aroud points to add, default 0.000135156253\\
    Returns Points
    '''
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]
    fin_diff_points=  torch.zeros((B,3,((2*D)+1)*N)).to(device).to(DTYPE)
    fin_diff_points[:,:,:N] = points.clone()

    i = 2
    if "X" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 0, stepsize)
        fin_diff_points[:,:,N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h

        i += 1

    
    if "Y" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 1, stepsize)
        fin_diff_points[:,:,(i-1)*N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h
        i += 1
    
    if "Z" in axis:
        points_h, points_neg_h = get_finite_diff_points(points, 2, stepsize)
        fin_diff_points[:,:,(i-1)*N:i*N] = points_h
        fin_diff_points[:,:,D*N+(i-1)*N:D*N+i*N] = points_neg_h
        i += 1
    
    return fin_diff_points

def gorkov_fin_diff(activations, points, axis="XYZ", stepsize = 0.000135156253,K1=None, K2=None,prop_function=propagate,prop_fun_args={}, board=TRANSDUCERS):
    '''
    Computes the Gorkov potential using finite differences to compute derivatives\\
    `activation` The transducer activations to use \\
    `points` The points to compute the potential at\\
    `axis` The axes to add points in as a string containing 'X', 'Y' and/or 'Z' eg 'XYZ' will use all three axis but 'YZ' will only add points in the YZ axis\\
    `stepsize` The distance aroud points to add, default 0.000135156253\\
    `K1` The value for K1 in the Gorkov equation, if None will use `c.V / (4*c.p_0*c.c_0**2)`
    `K2` The value for K2 in the Gorkov equation, if None will use `3*c.V / (4*(2*c.f**2 * c.p_0))`
    `prop_function` Function to use to compute pressure\\
    `prop_fun_args` Arguments to pass to `prop_function`\\
    `board` The transducer boards to use\\
    Returns gorkov potential at each point
    '''
    # torch.autograd.set_detect_anomaly(True)
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]

    
    if len(activations.shape) < 3:
        activations = torch.unsqueeze(activations,0).clone().to(device)

    fin_diff_points = get_finite_diff_points_all_axis(points, axis, stepsize)

    pressure_points = prop_function(activations, fin_diff_points,board=board,**prop_fun_args)
    # if len(pressure_points.shape)>1:
    # pressure_points = torch.squeeze(pressure_points,2)

    pressure = pressure_points[:,:N]
    pressure_fin_diff = pressure_points[:,N:]

    split = torch.reshape(pressure_fin_diff,(B,2, ((2*D))*N // 2))
    
    grad = (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    
    grad = torch.reshape(grad,(B,D,N))
    grad_abs_square = torch.pow(torch.abs(grad),2)
    grad_term = torch.sum(grad_abs_square,dim=1)

    if K1 is None:
        # K1 = 1/4 * c.V * (1/(c.c_0**2 * c.p_0) - 1/(c.c_p**2 * c.p_p))
        K1 = c.V / (4*c.p_0*c.c_0**2) #Assuming f1=f2=1
    if K2 is None:
        # K2 = 3/4 * c.V * ((c.p_0-c.p_p) / (c.f**2 * c.p_0 * (c.p_0+2*c.p_p)) )
        K2 = 3*c.V / (4*(2*c.f**2 * c.p_0)) #Assuming f1=f2=1
    
    # p_in =  torch.abs(pressure)
    p_in = torch.sqrt(torch.real(pressure) **2 + torch.imag(pressure)**2)
    if len(p_in.shape) > 2:
        p_in.squeeze_(2)
    # p_in = torch.squeeze(p_in,2)

    U = K1 * p_in**2 - K2 *grad_term
    
    return U

def gorkov_analytical(activations, points,board=TRANSDUCERS, axis="XYZ"):
    '''
    Computes the Gorkov potential using analytical derivative of the piston model\\
    `activation` The transducer activations to use \\
    `points` The points to compute the potential at\\
    `board` The transducer boards to use\\
    `axis` The axes to add points in as a string containing 'X', 'Y' and/or 'Z' eg 'XYZ' will use all three axis but 'YZ' will only add points in the YZ axis\\
    Returns gorkov potential at each point
    '''

    Fx, Fy, Fz = forward_model_grad(points)
    F = forward_model_batched(points,board)
    
    p = torch.abs(F@activations)**2
    
    if "X" in axis:
        grad_x = torch.abs((Fx@activations)**2)
    else:
        grad_x = 0
    
    if "Y" in axis:
        grad_y = torch.abs((Fy@activations)**2)
    else:
        grad_y = 0
   
    if "Z" in axis:
        grad_z = torch.abs((Fz@activations)**2)
    else:
        grad_z = 0


    K1 = c.V / (4*c.p_0*c.c_0**2)
    K2 = 3*c.V / (4*(2*c.f**2 * c.p_0))
    U = K1*p - K2*(grad_x+grad_y+grad_z)

    return U



if __name__ == "__main__":
    from acoustools.Utilities import create_points, forward_model
    from acoustools.Solvers import wgs_wrapper, wgs
    import matplotlib.pyplot as plt

    N=1
    B=1
    F_As = []
    F_FDs = []
    F_aFDs = []
    axis=0
    for _ in range(1):
        points = create_points(N,B)
        x = wgs_wrapper(points)
        # x = add_lev_sig(x)
        
        U_ag = gorkov_autograd(x,points)
        # F_aFD = force_fin_diff(x,points,U_function=gorkov_autograd).squeeze()
        # F_aFDs.append(F_aFD[axis].cpu().detach().numpy())

        U_fd = gorkov_fin_diff(x,points)
        # F = force_fin_diff(x,points).squeeze()
        # F_FDs.append(F[axis].cpu().detach().numpy())

        U_a = gorkov_analytical(x,points)
        # F_a = compute_force(x,points).squeeze()
        # F_As.append(F_a[axis].cpu().detach().numpy())

        # print(U_ag,U_fd,U_a, sep='\n')
        # print(F_aFD, F, F_a, sep='\n')
    # plt.scatter(F_FDs, F_aFDs)
    # plt.show()