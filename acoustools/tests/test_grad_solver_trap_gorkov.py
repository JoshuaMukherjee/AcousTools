if __name__ == '__main__':

    from acoustools.Solvers import gradient_descent_solver
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, propagate_abs
    from acoustools.Optimise.Objectives import gorkov_analytical_std_mean_objective
    from acoustools.Optimise.Constraints import constrain_phase_only
    from acoustools.Visualiser import Visualise

    import torch

    p = create_points(4,1, y=0)
    x = gradient_descent_solver(p,gorkov_analytical_std_mean_objective, 
                                    maximise=False, constrains=constrain_phase_only, 
                                    log=True, lr=1e2,iters=700)

    print(propagate_abs(x,p))
    
    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=p,vmax=5000)
