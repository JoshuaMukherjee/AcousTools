if __name__ == '__main__':
    from acoustools.Solvers import wgs
    from acoustools.Utilities import create_points, add_lev_sig, generate_pressure_targets, TOP_BOARD, device
    from acoustools.Optimise.Objectives import target_pressure_mse_objective, propagate_abs_sum_objective
    from acoustools.Optimise.Constraints import constrain_phase_only, constrant_normalise_amplitude
    from acoustools.Visualiser import Visualise, Visualise_mesh
    from acoustools.Mesh import load_multiple_scatterers
    from acoustools.BEM import propagate_BEM_pressure, compute_E

    import torch, os


    board = TOP_BOARD


    path = '../BEMMedia'
    paths = [path+"/bunny-lam2.stl"]
    scatterer = load_multiple_scatterers(paths, dzs=[-0.06], rotzs=[90])

    p = create_points(1,1, y=0,x=0,z=-0.01)

    E = compute_E(scatterer, p,board=board, path=path)

    x = wgs(p,A=E)

    A = torch.tensor((-0.03,0, 0))
    B = torch.tensor((0.03,0, 0))
    C = torch.tensor((-0.03,0, -0.06))
    normal = (0,1,0)
    origin = (0,0,0)


    Visualise(A,B,C, x, res=(300,300), points=p,vmax=5000,colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':scatterer,'board':board,'path':path}],clr_label="Pressure (Pa)")
