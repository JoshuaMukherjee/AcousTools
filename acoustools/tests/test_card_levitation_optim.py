if __name__ == '__main__':



    from acoustools.Mesh import load_scatterer, scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas,\
        get_centre_of_mass_as_points, get_weight, load_multiple_scatterers, merge_scatterers, get_lines_from_plane,get_plane, scatterer_file_name, translate
    from acoustools.Utilities import TRANSDUCERS, write_to_file, get_rows_in, propagate_abs, device, DTYPE
    from acoustools.BEM import grad_H, get_cache_or_compute_H_gradients, get_cache_or_compute_H,propagate_BEM_pressure, get_cache_or_compute_H_2_gradients
    from acoustools.Visualiser import Visualise, force_quiver, force_quiver_3d, Visualise_mesh
    from acoustools.Solvers import gradient_descent_solver, wgs
    from acoustools.Optimise.Constraints import constrain_phase_only
    from acoustools.Force import force_mesh, get_force_mesh_along_axis,torque_mesh
    import acoustools.Constants as Constants
    from acoustools.Levitator import LevitatorController

    import torch, vedo
    import matplotlib.pyplot as plt

    def get_H_for_fin_diffs(start,end, scatterers, board, steps=1, path="Media",print_lines=False, use_cache=True):
        direction = (end - start) / steps  

        translate(scatterers[0], start[0].item() - direction[0].item(), start[1].item() - direction[1].item(), start[2].item() - direction[2].item())
        scatterer = merge_scatterers(*scatterers)
        
        Hs = []
        Hxs = []
        Hys = []
        Hzs = []
        
        for i in range(steps+1):
            if print_lines:
                print(i)
            
            
            translate(scatterers[0], direction[0].item(), direction[1].item(), direction[2].item())
            scatterer = merge_scatterers(*scatterers)

            H = get_cache_or_compute_H(scatterer, board, path=path, print_lines=print_lines, use_cache_H=use_cache)
            Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board, path=path, print_lines=print_lines, use_cache_H_grad=use_cache)
        
            Hs.append(H)
            Hxs.append(Hx)
            Hys.append(Hy)
            Hzs.append(Hz)
        
        return Hs, Hxs, Hys, Hzs


    path = "../BEMMedia/"
    scatterer = load_scatterer('Card-lam2.stl',root_path=path, rotx=90)
    # Visualise_mesh(card,buffer_z=0.01)
    # exit()

    diff = 0.0025
    board = TRANSDUCERS

    scatterer_cells = get_centres_as_points(scatterer)
    
    mask = get_rows_in(scatterer_cells,scatterer_cells, expand=False)
    
    # scale_to_diameter(scatterer,0.04)

    print(scatterer)

    centres = get_centres_as_points(scatterer)
    centre_of_mass = get_centre_of_mass_as_points(scatterer)
 
    norms = get_normals_as_points(scatterer)
    
    areas = get_areas(scatterer)
    areas = areas.expand((1,-1,-1))

    Hx, Hy, Hz = get_cache_or_compute_H_gradients(scatterer, board,print_lines=True, path=path)
    H = get_cache_or_compute_H(scatterer,board,print_lines=True, path=path)

    # indexes = get_indexes_subsample(1700, centres)
 
    # weight = -1*0.0027*9.81
    # weight = -1*get_weight(ball)
    weight = -1*(0.1/1000)*9.81 #Measured value
    # weight = -1*0.002*9.81

    Hss = []
    Hxss = []
    Hyss = []
    Hzss = []

    SCALE = 10
    startX = torch.tensor([[-1*diff],[0],[0]])/SCALE
    endX = torch.tensor([[diff],[0],[0]])/SCALE
    Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX, endX, [scatterer.clone()], board, steps=1, use_cache=True, print_lines=False, path=path)
    Hss.append(Hs)
    Hxss.append(Hxs)
    Hyss.append(Hys)
    Hzss.append(Hzs)

    startY = torch.tensor([[0],[-1*diff],[0]])/SCALE
    endY = torch.tensor([[0],[diff],[0]])/SCALE
    Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY, endY, [scatterer.clone()], board, steps=1, use_cache=True, print_lines=False, path=path)
    Hss.append(Hs)
    Hxss.append(Hxs)
    Hyss.append(Hys)
    Hzss.append(Hzs)

    startZ = torch.tensor([[0],[0],[-1*diff]])/SCALE
    endZ = torch.tensor([[0],[0],[diff]])/SCALE
    Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ, endZ, [scatterer.clone()], board, steps=1, use_cache=True, print_lines=False, path=path)
    Hss.append(Hs)
    Hxss.append(Hxs)
    Hyss.append(Hys)
    Hzss.append(Hzs)
    
    '''
    Set parameters for optimisation
    '''

    def levitation_balance_greater_grad_torque(force_x, force_y, force_z, weight, torque, **params):
        a,b,c,d = params["weights"]

        FxsX = params["FxsX"]
        FysY = params["FysY"]
        FzsZ = params["FzsZ"]

        net_x = torch.sum(force_x)**2
        net_y = torch.sum(force_y)**2
        net_z = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
        # counter_weight = ((torch.sum(force_z) + weight)**2).unsqueeze_(0)
        
        balance = a* (net_x + net_y + net_z)

        greater_weight = b*(weight - torch.sum(force_z) ).unsqueeze_(0)


        grad_X = FxsX[0] - FxsX[-1]
        grad_Y = FysY[0] - FysY[-1]
        grad_Z = FzsZ[0] - FzsZ[-1]
        gradient = -1 *c * (grad_X + grad_Y + grad_Z)

        min_torque = d*torch.sum(torque**2,dim=[1,2])

        return balance + greater_weight + gradient + min_torque

    params = {
        "scatterer":scatterer,
        "norms":norms,
        "areas":areas,
        "weight":weight,
        # "weight":-1*0.00100530964,
        "Hgrad":(Hx, Hy, Hz),
        "H":H,
        # "loss":levitation_balance_grad_torque_direction,
        "loss":levitation_balance_greater_grad_torque,
        "loss_params":{
            # 'weights':[40,5,1,50000] 
            'weights':[1,1,100,1] 
        },
        "indexes":mask.squeeze_(),
        "diff":diff,
        "scatterer_elements":[scatterer],
        "Hss":Hss,
        "Hxss":Hxss,
        "Hyss":Hyss,
        "Hzss":Hzss
    }


    BASE_LR = 1e-2
    MAX_LR = 1e-1
    EPOCHS = 200

    scheduler = torch.optim.lr_scheduler.CyclicLR
    scheduler_args = {
        "max_lr":MAX_LR,
        "base_lr":BASE_LR,
        "cycle_momentum":False,
        "step_size_up":25
    }
    # scheduler=scheduler, scheduler_args=scheduler_args    

    '''
    Initialise with WGS below COM - ignore all scattering etc as just a starting point
    '''
    
    p0 = (centre_of_mass[:,0,:].cpu().detach().item(),centre_of_mass[:,1,:].cpu().detach().item(),centre_of_mass[:,2,:].cpu().detach().item())
    p1 = (centre_of_mass[:,0,:].cpu().detach().item(),centre_of_mass[:,1,:].cpu().detach().item(),-100)
    cell = scatterer.find_cells_along_line(p0,p1,tol=1e-7)[0]
    below_COM = torch.tensor(scatterer.coordinates[scatterer.cells[cell][0]]).unsqueeze_(0).unsqueeze_(2).to(device).to(DTYPE)
    
    below_COM_1 = below_COM.clone()
    below_COM_1[:,0,:] -= 0.005

    below_COM_2 = below_COM.clone()
    below_COM_2[:,0,:] += 0.005

    below_COM = torch.cat([below_COM_1, below_COM_2], dim=2)

    x_start = wgs(below_COM,iter=5)

    '''
    Phase retrieval
    '''

    
    def BEM_levitation_objective_subsample_stability_fin_diff_single_element(transducer_phases, points, board, targets=None, **objective_params):
        ''' Expects an element of objective_params named:\\
        `norms` containing all normals in the same order as points \\
        `areas` containing the areas for the cell containing each point\\
        `scatterer` containing the mesh to optimise around\\
        `indexes` containing subsample to use
        `diff` containing difference to use in fin diffs
        `scatterer_elements` contains list of [object to levitate, walls]
        '''


        scatterer = objective_params["scatterer"]
        norms = objective_params["norms"]
        areas = objective_params["areas"]
        indexes = objective_params["indexes"]
        diff = objective_params["diff"]
        scatter_elems = objective_params["scatterer_elements"]

        loss_function = objective_params["loss"]
        
        if "loss_params" in objective_params:
            loss_params = objective_params["loss_params"]
        else:
            loss_params = {} 

        params = {
            "scatterer":scatterer
        }

        centre_of_mass = get_centre_of_mass_as_points(scatterer)

        if "Hgrad" not in objective_params:
            Hx, Hy, Hz = get_cache_or_compute_H_gradients(None, transducers=board, **{"scatterer":scatterer })
        else:
            Hx, Hy, Hz = objective_params["Hgrad"]
        
        if "H" not in objective_params:
            H = get_cache_or_compute_H(scatterer,board)
        else:
            H = objective_params["H"]

        # E = compute_E(scatterer,points,board, H=H)
        force = force_mesh(transducer_phases,points,norms,areas,board,grad_H,params,Ax=Hx, Ay=Hy, Az=Hz,F=H)
        # force = force_mesh(transducer_phases,points,norms,areas,board,BEM_forward_model_grad,params,F=E)
        torque = torque_mesh(transducer_phases,points,norms,areas,centre_of_mass,board,force=force)
    
        startX = torch.tensor([[-1*diff],[0],[0]]) 
        endX = torch.tensor([[diff],[0],[0]]) 

        startY = torch.tensor([[0],[-1*diff],[0]])
        endY = torch.tensor([[0],[diff],[0]])

        startZ = torch.tensor([[0],[0],[-1*diff]])
        endZ = torch.tensor([[0],[0],[diff]])
        
        ball = scatter_elems[0]

        if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
            Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startX/1000, endX/1000, [ball.clone()], board, steps=1, use_cache=True, print_lines=False, path=path)
        else:
            Hs = objective_params["Hss"][0]
            Hxs = objective_params["Hxss"][0]
            Hys = objective_params["Hyss"][0]
            Hzs = objective_params["Hzss"][0]

        FxsX, _, _ = get_force_mesh_along_axis(startX/1000, endX/1000, transducer_phases, [ball.clone()], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs, path=path)
        
        if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
            Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startY/1000, endY/1000, [ball.clone()], board, steps=1, use_cache=True, print_lines=False, path=path)
        else:
            Hs = objective_params["Hss"][1]
            Hxs = objective_params["Hxss"][1]
            Hys = objective_params["Hyss"][1]
            Hzs = objective_params["Hzss"][1]

        _, FysY, _ = get_force_mesh_along_axis(startY/1000, endY/1000, transducer_phases, [ball.clone()], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs, path=path)
        
        if "Hss" not in objective_params or "Hxss" not in objective_params or "Hyss" not in objective_params or "Hzss" not in objective_params:
            Hs, Hxs, Hys, Hzs = get_H_for_fin_diffs(startZ/1000, endZ/1000, [ball.clone()], board, steps=1, use_cache=True, print_lines=False, path=path)
        else:
            Hs = objective_params["Hss"][2]
            Hxs = objective_params["Hxss"][2]
            Hys = objective_params["Hyss"][2]
            Hzs = objective_params["Hzss"][2]

        _, _, FzsZ = get_force_mesh_along_axis(startZ/1000, endZ/1000, transducer_phases, [ball.clone()], board,indexes,steps=1, use_cache=True, print_lines=False, Hs=Hs, Hxs = Hxs, Hys=Hys, Hzs=Hzs, path=path)



        if "weight" in objective_params:
            weight = objective_params["weight"]
        else:
            weight = -1*get_weight(scatterer)

        force_x = force[:,0,:][:,indexes]
        force_y = force[:,1,:][:,indexes]
        force_z = force[:,2,:][:,indexes]
        torque = torque[:,:,indexes]

        loss_params["FxsX"]=FxsX
        loss_params["FysY"]=FysY
        loss_params["FzsZ"]=FzsZ

        
        return loss_function(force_x, force_y, force_z, weight, torque, **loss_params)

    save_set_n = [n-1 for n in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,100,150,200]]
    # save_set_n = [n-1 for n in [1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,100]]
    x, loss, result = gradient_descent_solver(centres, BEM_levitation_objective_subsample_stability_fin_diff_single_element,constrains=constrain_phase_only,objective_params=params,log=True,\
                                iters=EPOCHS,lr=BASE_LR, optimiser=torch.optim.Adam, board=board,scheduler=scheduler, scheduler_args=scheduler_args, return_loss=True, save_set_n=save_set_n )
    
    centres = get_centres_as_points(scatterer)
    pressures = propagate_BEM_pressure(x,centres,H=H, scatterer=scatterer,board=TRANSDUCERS,path=path)
    print(pressures.max())
    Visualise_mesh(scatterer, pressures,equalise_axis=True)

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x)

    lev = LevitatorController(ids=(73,53))
    lev.levitate(x)
    print('Levitating...')
    input()
    print('Stopping...')
    lev.disconnect()
    print('Stopped')