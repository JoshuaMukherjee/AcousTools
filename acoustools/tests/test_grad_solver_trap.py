if __name__ == '__main__':

    from acoustools.Solvers import gradient_descent_solver
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise

    import torch


    

    p = create_points(4,1, y=0)
    x = gradient_descent_solver(p,propagate_abs_sum_objective, 
                                    maximise=False, constrains=constrain_phase_only, log=False, lr=1e1,iters=5000, 
                                    scheduler=torch.optim.lr_scheduler.CyclicLR,scheduler_args={'base_lr':1e1,'max_lr':1e2,'cycle_momentum':False,'step_size_up':100})

    
    # targets = generate_pressure_targets(4,1,max_val=4000,min_val=4000).squeeze_(2)
    # x = gradient_descent_solver(p,target_pressure_mse_objective, 
    #                                 maximise=False, constrains=constrain_phase_only, lr=1e-1, iters=500, targets=targets)

    x = add_lev_sig(x)

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=p,vmax=5000)
