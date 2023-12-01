

if __name__ == "__main__":
    from acoustools.Solvers import gradient_descent_solver
    from acoustools.Utilities import create_points, propagate_abs
    from acoustools.Optimise.Objectives import propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only
        

    p = create_points(4,2)

    x = gradient_descent_solver(p,propagate_abs_sum_objective, 
                                maximise=True, constrains=constrain_phase_only, log=True, lr=1e-1)

    print(propagate_abs(x,p))
