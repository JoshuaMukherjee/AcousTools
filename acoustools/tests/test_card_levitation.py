if __name__ == '__main__':


    from acoustools.Utilities import create_points, propagate_abs, TOP_BOARD, BOTTOM_BOARD
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise_single, Visualise
    from acoustools.Constants import wavelength

    import torch
    import matplotlib.pyplot as plt


    X = wavelength
    Y = 0.09
    Z = 0.02
    ps = []
    for x in [-1,1]:
        for y in [-1,1]:
            for z in [-1,1]:
                p = create_points(1,1,x=x*X, y=y*Y, z=z*Z)
                ps.append(p)
    

    ps = torch.cat(ps,dim=2)

    x = wgs(ps)


    A = torch.tensor((-0.09,Y, 0.09))
    B = torch.tensor((0.09,Y, 0.09))
    C = torch.tensor((-0.09,Y, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=ps)


    # top = Visualise_single(A,B,C, x[:,256:],propagate_abs,{'board':TOP_BOARD})
    # bottom = Visualise_single(A,B,C, x[:,:256],propagate_abs,{'board':BOTTOM_BOARD})

    # ax1 = plt.subplot(1,2,1)
    # ax1.matshow(top, cmap='hot')
    # ax2 = plt.subplot(1,2,2)
    # ax2.matshow(bottom, cmap='hot')
    # plt.show()


    



