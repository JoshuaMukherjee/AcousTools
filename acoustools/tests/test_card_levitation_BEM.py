if __name__ == '__main__':


    from acoustools.Utilities import create_points, propagate_abs, TOP_BOARD, BOTTOM_BOARD, TRANSDUCERS
    from acoustools.Solvers import wgs, gradient_descent_solver
    from acoustools.Visualiser import Visualise_single, Visualise, Visualise_mesh
    from acoustools.Constants import wavelength
    from acoustools.Mesh import load_scatterer, get_lines_from_plane, get_centres_as_points
    from acoustools.BEM import compute_E, propagate_BEM_pressure
    from acoustools.Levitator import LevitatorController

    import torch, vedo
    import matplotlib.pyplot as plt

    path = "../BEMMedia/"

    card = load_scatterer('Card-lam2.stl',root_path=path, rotx=90)
    # Visualise_mesh(card,buffer_z=0.01)
    # exit()


    X = 0.02
    Y = 0.09
    Z = 0.004
    ps = []
    for x in [-1,1]:
        for y in [-1,1]:
            for z in [-1,1]:
                p = create_points(1,1,x=x*X, y=y*Y, z=z*Z)
                ps.append(p)
    

    ps = torch.cat(ps,dim=2)
    print(ps.requires_grad)

    E,F,G,H = compute_E(card, ps,return_components=True,path=path, board=TRANSDUCERS)
    x = wgs(ps,A=E)
    print(torch.abs(E@x))



    # A = torch.tensor((-0.09,Y, 0.09))
    # B = torch.tensor((0.09,Y, 0.09))
    # C = torch.tensor((-0.09,Y, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)
    # Visualise(A,B,C,x,ps)
    # exit()

    lev = LevitatorController(ids=(73,53))
    lev.levitate(x)
    print('Levitating...')
    input()
    print('Stopping...')
    lev.disconnect()
    print('Stopped')


    # centres = get_centres_as_points(card)
    # pressures = propagate_BEM_pressure(x,centres,H=H, scatterer=card,board=TRANSDUCERS)
    # print(pressures.max())
    # Visualise_mesh(card, pressures,equalise_axis=True)


    # A = torch.tensor((-0.09,Y, 0.09))
    # B = torch.tensor((0.09,Y, 0.09))
    # C = torch.tensor((-0.09,Y, -0.09))
    # normal = (0,1,0)
    # origin = (0,0,0)

    # Visualise(A,B,C, x, points=ps, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':card,'H':H}],add_lines_functions=[get_lines_from_plane],add_line_args=[{'scatterer':card,'origin':origin,'normal':normal}])


    # top = Visualise_single(A,B,C, x[:,256:],propagate_abs,{'board':TOP_BOARD})
    # bottom = Visualise_single(A,B,C, x[:,:256],propagate_abs,{'board':BOTTOM_BOARD})

    # ax1 = plt.subplot(1,2,1)
    # ax1.matshow(top, cmap='hot')
    # ax2 = plt.subplot(1,2,2)
    # ax2.matshow(bottom, cmap='hot')
    # plt.show()


    



