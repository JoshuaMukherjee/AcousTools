

if __name__ == "__main__":
    from acoustools.Solvers import gradient_descent_solver
    from acoustools.Utilities import create_points, propagate_abs, add_lev_sig
    from acoustools.Optimise.Objectives import propagate_abs_sum_objective, gorkov_analytical_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only
    from acoustools.Gorkov import gorkov_analytical

    p = create_points(4,2)

    x = gradient_descent_solver(p,propagate_abs_sum_objective, 
                                maximise=True, constrains=constrain_phase_only, log=False, lr=1e-1)

    print(propagate_abs(x,p))


    x2 = gradient_descent_solver(p,gorkov_analytical_sum_objective, constrains=constrain_phase_only,log=False, lr=1e-1)

    print(propagate_abs(x2,p))
    x2 = add_lev_sig(x2)
    print(gorkov_analytical(x2,p))

