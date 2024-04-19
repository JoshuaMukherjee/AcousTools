if __name__ == "__main__":
    from acoustools.Solvers import gradient_descent_solver
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig, generate_pressure_targets, generate_gorkov_targets
    from acoustools.Optimise.Objectives import propagate_abs_sum_objective, gorkov_analytical_sum_objective, pressure_abs_gorkov_trapping_stiffness_objective, target_pressure_mse_objective, target_gorkov_mse_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Gorkov import gorkov_analytical
    from acoustools.Visualiser import Visualise

    import torch    

    def test_pressure():
        p = create_points(4,2)
        x = gradient_descent_solver(p,propagate_abs_sum_objective, 
                                    maximise=True, constrains=constrain_phase_only, log=False, lr=1e-1)

        print(propagate_abs(x,p))

    def test_gorkov():
        p = create_points(4,2)
        x2 = gradient_descent_solver(p,gorkov_analytical_sum_objective, constrains=constrain_phase_only,log=False, lr=1e-1)

        print(propagate_abs(x2,p))
        x2 = add_lev_sig(x2)
        print(gorkov_analytical(x2,p))

    def test_gorkov_trapping():
        p = create_points(4,2)
        x3 = gradient_descent_solver(p,pressure_abs_gorkov_trapping_stiffness_objective, 
                                    maximise=True, constrains=constrain_phase_only, lr=1e-1, iters=200)

        print(propagate_abs(x3,p))
        x3 = add_lev_sig(x3)
        print(gorkov_analytical(x3,p))

    def test_pressure_target():
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy 

        N = 4
        B = 100

        p = create_points(N,B)
        MIN = 1000
        MAX = 5000
        targets = generate_pressure_targets(N,B,max_val=MAX, min_val=MIN).squeeze_(2)
        x4 = gradient_descent_solver(p,target_pressure_mse_objective, 
                                    maximise=False, constrains=constrain_phase_only, lr=1e-1, iters=5000, targets=targets, log=True)
        
        # print(targets)
        # print(propagate_abs(x4,p))

        xs = targets.squeeze_().cpu().flatten().detach().numpy()
        ys = propagate_abs(x4, p).squeeze_().cpu().flatten().detach().numpy()

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xs, ys)
        print('R^2:',r_value**2)

    
        plt.scatter(xs,ys)
        plt.xlim((MIN-200, MAX+200))
        plt.ylim((MIN-200, MAX+200))
        plt.plot([np.min(xs),np.max(xs)],[np.min(xs),np.max(xs)],color="red")
        plt.xlabel("Target (Pa)")
        plt.ylabel("Output (Pa)")
        plt.show()
    
    def test_gorkov_target():
        import matplotlib.pyplot as plt
        import numpy as np
        import scipy 

        N = 4
        B = 100
        p = create_points(N,B)
        targets_u = generate_gorkov_targets(N,B,min_val=-1e-5,max_val=-1e-6)
        x5 = gradient_descent_solver(p,target_gorkov_mse_objective, 
                                     constrains=constrain_phase_only, lr=1e3, iters=5000, targets=targets_u,log=True,
                                     objective_params={"no_sig":True})

        # x5 = add_lev_sig(x5)
        
        xs = targets_u.squeeze_().cpu().flatten().detach().numpy()
        ys = gorkov_analytical(x5, p).squeeze_().cpu().flatten().detach().numpy()

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(xs, ys)
        print('R^2:',r_value**2)

        plt.scatter(xs,ys)
        # plt.xlim((-1e-6, -1e-7))
        # plt.ylim((-1e-6, -1e-7))
        plt.xlabel("Target")
        plt.ylabel("Output")
        plt.plot([np.min(xs),np.max(xs)],[np.min(xs),np.max(xs)],color="red")
        plt.show()

    def test_gorkov_pressure_traps():

        def gorkov_pressure_target(transducer_phases, points, board, targets, **objective_params):
            pressure_point = points[:,:,0].unsqueeze(2)
            pressure= torch.sum(propagate_abs(transducer_phases,pressure_point,board),dim=1)

            gorkov_point =  points[:,:,1].unsqueeze(2)
            U = gorkov_analytical(transducer_phases, gorkov_point, board, 'XYZ')

            return -1*pressure + 3e8*U.squeeze()

        p = create_points(2,1,y=0)
        x = gradient_descent_solver(p,gorkov_pressure_target, 
                                    constrains=constrain_phase_only, log=True, lr=1e-1,iters=1000)

        print(propagate_abs(x,p))
        
        
        A = torch.tensor((-0.09,0, 0.09))
        B = torch.tensor((0.09,0, 0.09))
        C = torch.tensor((-0.09,0, -0.09))
        normal = (0,1,0)
        origin = (0,0,0)

        Visualise(A,B,C, x, points=p)



    test_gorkov_pressure_traps()
    