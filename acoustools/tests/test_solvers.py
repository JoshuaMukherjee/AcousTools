if __name__ == '__main__':

    from acoustools.Solvers import naive_solver_wrapper, wgs_wrapper, gspat_wrapper
    from acoustools.Utilities import create_points, propagate_abs

    import matplotlib.pyplot as plt

    import torch

    pressures_naive = []
    pressures_wgs = []
    pressure_gspat = []

    std_naive = []
    std_wgs = []
    std_gspat = []

    for i in range(100):

        p = create_points(4,1)
        
        x_n = naive_solver_wrapper(p)
        x_wgs = wgs_wrapper(p)
        x_gspat = gspat_wrapper(p)

        p_n = propagate_abs(x_n, p)
        p_wgs = propagate_abs(x_wgs, p)
        p_gspat = propagate_abs(x_gspat, p)

        pressures_naive.append(torch.mean(p_n).cpu().detach())
        pressures_wgs.append(torch.mean(p_wgs).cpu().detach())
        pressure_gspat.append(torch.mean(p_gspat).cpu().detach())

        std_naive.append(torch.std(p_n).cpu().detach())
        std_wgs.append(torch.std(p_wgs).cpu().detach())
        std_gspat.append(torch.std(p_gspat).cpu().detach())
    


    to_plot_mean = {
        'WGS':pressures_wgs,
        'GSPAT':pressure_gspat,
        'Niave':pressures_naive
    }

    to_plot_std = {
        'WGS':std_wgs,
        'GSPAT':std_gspat,
        'Niave':std_naive
    }


    plt.subplot(1,2,1)
    plt.boxplot(to_plot_mean.values())
    plt.xticks(ticks=[1,2,3],labels=to_plot_mean.keys())
    plt.title('Mean Pressure Between Sets of 4 Points')
    plt.ylabel('Pressure (Pa)')
    plt.tight_layout()

    plt.subplot(1,2,2)
    plt.boxplot(to_plot_std.values())
    plt.xticks(ticks=[1,2,3],labels=to_plot_std.keys())
    plt.title('Standard Deviation of Pressure Between Points in Sets of 4 Points')
    plt.ylabel('Pressure (Pa)')
    plt.tight_layout()

    plt.show()
