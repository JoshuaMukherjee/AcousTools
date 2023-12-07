import torch
from acoustools.Utilities import device, propagate, propagate_abs, add_lev_sig, forward_model_batched, forward_model_grad, TRANSDUCERS, forward_model_second_derivative_unmixed,forward_model_second_derivative_mixed, return_matrix
import acoustools.Constants as c


def gorkov_autograd(activation, points, K1=None, K2=None, retain_graph=False):

    var_points = torch.autograd.Variable(points.data, requires_grad=True).to(device)

    B = points.shape[0]
    N = points.shape[2]
    
    if len(activation.shape) < 3:
        activation.unsqueeze_(0)    
    

    pressure = propagate(activation,var_points)

    if B > 1:
        if N > 1:
            grad_pos = torch.autograd.grad(pressure, var_points,grad_outputs=torch.ones((B,N),device=device)+1j, retain_graph=retain_graph)[0]
        else:
            grad_pos = torch.autograd.grad(pressure, var_points)[0]
    else:
        if N > 1:
            grad_pos = torch.autograd.grad(pressure, var_points,grad_outputs=torch.ones((N),device=device)+1j, retain_graph=retain_graph)[0]
        else:
            grad_pos = torch.autograd.grad(pressure, var_points)[0]
    
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

def gorkov_fin_diff(activations, points, axis="XYZ", stepsize = 0.000135156253,K1=None, K2=None,prop_function=propagate,prop_fun_args={}):
    # torch.autograd.set_detect_anomaly(True)
    B = points.shape[0]
    D = len(axis)
    N = points.shape[2]
    fin_diff_points=  torch.zeros((B,3,((2*D)+1)*N)).to(device)
    fin_diff_points[:,:,:N] = points.clone()
    
    
    if len(activations.shape) < 3:
        activations = torch.unsqueeze(activations,0).clone().to(device)

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


    pressure_points = prop_function(activations, fin_diff_points,**prop_fun_args)
    pressure_points = torch.squeeze(pressure_points,2)

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
    
    F = forward_model_batched(points)
    Fx, Fy, Fz = forward_model_grad(points)
    Fxx, Fyy, Fzz = forward_model_second_derivative_unmixed(points)
    Fxy, Fxz, Fyz = forward_model_second_derivative_mixed(points)

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


    return torch.real(force)

def torque_mesh(activations, points, norms, areas, centre_of_mass, board,force=None, grad_function=forward_model_grad,grad_function_args={}):
    
    if force is None:
        force = force_mesh(activations, points, norms, areas, board,grad_function,grad_function_args)
    

    displacement = points - centre_of_mass
    displacement = displacement.to(torch.float64)


    torque = torch.linalg.cross(displacement,force,dim=1)

    return torch.real(torque)


if __name__ == "__main__":
    from acoustools.Utilities import create_points, forward_model
    from acoustools.Solvers import wgs_wrapper, wgs

    from acoustools.Visualiser import Visualise


    points = create_points(4,1,x=0)
    x = wgs_wrapper(points)
    x = add_lev_sig(x)

    A = torch.tensor((0,-0.07, 0.07))
    B = torch.tensor((0,0.07, 0.07))
    C = torch.tensor((0,-0.07, -0.07))


    res = (200,200)

    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])
    step_x = AB / res[0]
    step_y = AC / res[1]

    positions = torch.zeros((1,3,res[0]*res[1])).to(device)

    for i in range(0,res[0]):
        for j in range(res[1]):
            positions[:,:,i*res[0]+j] = A + step_x * i + step_y * j

    print("Computing Force...")
    fx, fy, fz = compute_force(x,positions, return_components=True)


    fx = torch.reshape(fx, res)
    fy = torch.reshape(fy, res)
    fz = torch.reshape(fz, res)

    fx = torch.rot90(torch.fliplr(fx))
    fy = torch.rot90(torch.fliplr(fy))
    fz = torch.rot90(torch.fliplr(fz))
    print("Plotting...")

    Visualise(A,B,C,x,colour_functions=None,points=points,res=res,vmin=-3e-4, vmax= 1e-4, matricies=[fy,fz])

    # points = create_points(4,1)
    # x = wgs_wrapper(points)
    # x = add_lev_sig(x)
    # force = get_force_axis(x,points)
    # print(force)
    
    # def run():
    #     N =4
    #     B=2
    #     points=  create_points(N,B=B)
    #     print(points)
    #     xs = torch.zeros((B,512,1)) +0j
    #     for i in range(B):
    #         A = forward_model(points[i,:]).to(device)
    #         _, _, x = wgs(A,torch.ones(N,1).to(device)+0j,200)
    #         xs[i,:] = x


    #     xs = add_lev_sig(xs)

    #     gorkov_AG = gorkov_autograd(xs,points)
    #     print(gorkov_AG)

    #     gorkov_FD = gorkov_fin_diff(xs,points,axis="XYZ")
    #     print(gorkov_FD)
    # run()