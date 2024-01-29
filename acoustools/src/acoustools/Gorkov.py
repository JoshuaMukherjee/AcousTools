import torch
from acoustools.Utilities import device, propagate, propagate_abs, add_lev_sig, forward_model_batched, forward_model_grad, TRANSDUCERS, forward_model_second_derivative_unmixed,forward_model_second_derivative_mixed, return_matrix
import acoustools.Constants as c
from acoustools.BEM import grad_2_H, grad_H, get_cache_or_compute_H, get_cache_or_compute_H_gradients
from acoustools.Mesh import translate, get_centre_of_mass_as_points, get_centres_as_points, get_normals_as_points, get_areas, merge_scatterers

def gorkov_autograd(activation, points, K1=None, K2=None, retain_graph=False,**params):

    var_points = torch.autograd.Variable(points.data, requires_grad=True).to(device).to(torch.complex64)

    B = points.shape[0]
    N = points.shape[2]
    
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)    
    
    pressure = propagate(activation.to(torch.complex64),var_points)
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
    #points = Bx3x4
    points_h = points.clone()
    points_neg_h = points.clone()
    points_h[:,axis,:] = points_h[:,axis,:] + stepsize
    points_neg_h[:,axis,:] = points_neg_h[:,axis,:] - stepsize

    return points_h, points_neg_h

def get_finite_diff_points_all_axis(points,axis="XYZ", stepsize = 0.000135156253):
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]
    fin_diff_points=  torch.zeros((B,3,((2*D)+1)*N)).to(device).to(torch.complex64)
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

def gorkov_fin_diff(activations, points, axis="XYZ", stepsize = 0.000135156253,K1=None, K2=None,prop_function=propagate,prop_fun_args={}):
    # torch.autograd.set_detect_anomaly(True)
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]

    
    if len(activations.shape) < 3:
        activations = torch.unsqueeze(activations,0).clone().to(device)

    fin_diff_points = get_finite_diff_points_all_axis(points, axis, stepsize)

    pressure_points = prop_function(activations, fin_diff_points,**prop_fun_args)
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
    # p_in = torch.squeeze(p_in,2)

    U = K1 * p_in**2 - K2 *grad_term
    
    return U

def force_fin_diff(activations, points, axis="XYZ", stepsize = 0.000135156253,K1=None, K2=None,U_function=gorkov_fin_diff,U_fun_args={}):
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]

    fin_diff_points = get_finite_diff_points_all_axis(points, axis, stepsize)
    
    U_points = U_function(activations, fin_diff_points, axis=axis, stepsize=stepsize/10 ,K1=K1,K2=K2,**U_fun_args)
    U_grads = U_points[:,N:]
    split = torch.reshape(U_grads,(B,2, ((2*D))*N // 2))
    
    F =  (split[:,0,:] - split[:,1,:]) / (2*stepsize)
    return F

def gorkov_analytical(activations, points,board=TRANSDUCERS, axis="XYZ"):
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

def compute_force(activations, points,board=TRANSDUCERS,return_components=False):
    
    F = forward_model_batched(points,transducers=board)
    Fx, Fy, Fz = forward_model_grad(points,transducers=board)
    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points,transducers=board)
    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points,transducers=board)

    p   = torch.abs(F@activations)
    Px  = torch.abs(Fx@activations)
    Py  = torch.abs(Fy@activations)
    Pz  = torch.abs(Fz@activations)
    Pxx = torch.abs(Fxx@activations)
    Pyy = torch.abs(Fyy@activations)
    Pzz = torch.abs(Fzz@activations)
    Pxy = torch.abs(Fxy@activations)
    Pxz = torch.abs(Fxz@activations)
    Pyz = torch.abs(Fyz@activations)


    
    K1 = c.V / (4*c.p_0*c.c_0**2)
    K2 = 3*c.V / (4*(2*c.f**2 * c.p_0))

    single_sum = 2*K2*(Pz+Py+Pz)

    force_x = -1 * (2*p * (K1 * Px - K2*(Pxz+Pxy+Pxx)) - Px*single_sum)
    force_y = -1 * (2*p * (K1 * Py - K2*(Pyz+Pyy+Pxy)) - Py*single_sum)
    force_z = -1 * (2*p * (K1 * Pz - K2*(Pzz+Pyz+Pxz)) - Pz*single_sum)


    if return_components:
        return force_x, force_y, force_z
    else:
        force = torch.cat([force_x, force_y, force_z],2)
        return force

def get_force_axis(activations, points,board=TRANSDUCERS, axis=2):
    forces = compute_force(activations, points,return_components=True)
    force = forces[axis]

    return force

def force_mesh(activations, points, norms, areas, board, grad_function=forward_model_grad, grad_function_args={},F=None, Ax=None, Ay=None,Az=None):
    
    p = propagate(activations,points,board,A=F)
    pressure = torch.abs(p)**2
    
    if Ax is None or Ay is None or Az is None:
        Ax, Ay, Az = grad_function(points=points, transducers=board, **grad_function_args)
    
    px = (Ax@activations).squeeze_(2).unsqueeze_(0)
    py = (Ay@activations).squeeze_(2).unsqueeze_(0)
    pz = (Az@activations).squeeze_(2).unsqueeze_(0)

    px[px.isnan()] = 0
    py[py.isnan()] = 0
    pz[pz.isnan()] = 0


    grad = torch.cat((px,py,pz),dim=1).to(torch.complex128)
    grad_norm = torch.norm(grad,2,dim=1)**2

    
    k1 = 1/ (2*c.p_0*(c.c_0**2))
    k2 = 1/ (c.k**2)

    pressure = torch.unsqueeze(pressure,1).expand(-1,3,-1)


    force = (k1 * (pressure * norms - k2*grad_norm*norms)) * areas
    force = torch.real(force) #Im(F) == 0 but needs to be complex till now for dtype compatability

    # print(torch.sgn(torch.sgn(force) * torch.log(torch.abs(force))) == torch.sgn(force))

    return force

def torque_mesh(activations, points, norms, areas, centre_of_mass, board,force=None, grad_function=forward_model_grad,grad_function_args={},F=None, Ax=None, Ay=None,Az=None):
    
    if force is None:
        force = force_mesh(activations, points, norms, areas, board,grad_function,grad_function_args,F=F, Ax=Ax, Ay=Ay, Az=Az)
    
    displacement = points - centre_of_mass
    displacement = displacement.to(torch.float64)

    torque = torch.linalg.cross(displacement,force,dim=1)

    return torch.real(torque)

def force_mesh_derivative(activations, points, norms, areas, board, scatterer,Hx = None, Hy=None, Hz=None, Haa=None):
    print("Warning probably not correct...")
    if Hx is None or Hy is None or Hz is None:
        Hx, Hy, Hz, A, A_inv, Ax, Ay, Az = grad_H(points, scatterer, board, True)
    else:
        A, A_inv, Ax, Ay, Az = None, None, None, None, None

    if Haa is None:
        Haa = grad_2_H(points, scatterer, board, A, A_inv, Ax, Ay, Az)
    
    Ha = torch.stack([Hx,Hy,Hz],dim=1)

    Pa = Ha@activations
    Paa = Haa@activations

    Pa = Pa.squeeze(3)
    Paa = Paa.squeeze(3)

    k1 = 1/ (2*c.p_0*(c.c_0**2))
    k2 = 1/ (c.k**2)


    Faa =areas * k1 * (Pa * norms - 2*k2*norms*Pa*Paa)

    return Faa

def get_force_mesh_along_axis(start,end, activations, scatterers, board, mask=None, steps=200, path="Media",print_lines=False, use_cache=True, Hs = None, Hxs=None, Hys=None, Hzs=None):
    '''
    First element in scatterers is the mesh to levitate, rest is considered reflectors
    '''
    # if Ax is None or Ay is None or Az is None:
    #     Ax, Ay, Az = grad_function(points=points, transducers=board, **grad_function_args)
    direction = (end - start) / steps

    translate(scatterers[0], start[0].item() - direction[0].item(), start[1].item() - direction[1].item(), start[2].item() - direction[2].item())
    scatterer = merge_scatterers(*scatterers)

    points = get_centres_as_points(scatterer)
    if mask is None:
        mask = torch.ones(points.shape[2]).to(bool)

    Fxs = []
    Fys = []
    Fzs = []

    for i in range(steps+1):
        if print_lines:
            print(i)
        
        
        translate(scatterers[0], direction[0].item(), direction[1].item(), direction[2].item())
        scatterer = merge_scatterers(*scatterers)

        points = get_centres_as_points(scatterer)
        areas = get_areas(scatterer)
        norms = get_normals_as_points(scatterer)

        if Hs is None:
            H = get_cache_or_compute_H(scatterer, board, path=path, print_lines=print_lines, use_cache_H=use_cache)
        else:
            H = Hs[i]
        
        if Hxs is None or Hys is None or Hzs is None:
            Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board, path=path, print_lines=print_lines, use_cache_H_grad=use_cache)
        else:
            Hx = Hxs[i]
            Hy = Hys[i]
            Hz = Hzs[i]
        

        force = force_mesh(activations, points, norms, areas, board, F=H, Ax=Hx, Ay=Hy, Az=Hz)

        force = torch.sum(force[:,:,mask],dim=2).squeeze()
        Fxs.append(force[0])
        Fys.append(force[1])
        Fzs.append(force[2])
        
        # print(i, force[0].item(), force[1].item(),force[2].item())
    return Fxs, Fys, Fzs


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