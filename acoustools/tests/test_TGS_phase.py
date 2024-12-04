if __name__ == '__main__':

    from acoustools.Utilities import create_points, forward_model_batched, device, TRANSDUCERS, propagate
    from acoustools.Solvers import wgs, temporal_wgs

    import torch
    import matplotlib.pyplot as plt

    N = 2
    M=1000
    pi = torch.pi
    
    point = create_points(N,1)
    A=forward_model_batched(point).to(device)
    x_wgs = wgs(point, A=A)
    z_wgs = A@x_wgs

    x_temp = x_wgs
    z_temp = z_wgs

    temp_phases = [torch.mean(torch.angle(x_temp))]
    wgs_phases = [torch.mean(torch.angle(x_wgs))]

    wgs_pressure = []
    temp_pressure = []
            
    T_in = pi/64
    T_out = 0

    for i in range(M):
        point += torch.rand_like(point)/10000
        

        A=forward_model_batched(point).to(device)

        _, _, x_temp = temporal_wgs(A,torch.ones(N,1).to(device)+0j,200,x_temp,z_temp,T_in,T_out)
        z_temp = A@x_temp
        # print(torch.abs(z_temp))
        temp_phases.append(torch.mean(torch.angle(x_temp)))
        temp_pressure.append(torch.mean(torch.abs(z_temp)))


        x_wgs =  wgs(point, A=A)
        wgs_phases.append(torch.mean(torch.angle(x_wgs)))
        wgs_pressure.append(torch.mean(torch.abs(A@x_wgs)))
        
    temp_phases = [torch.abs(torch.atan2(torch.sin(i),torch.cos(i))) for i in temp_phases]
    wgs_phases = [torch.abs(torch.atan2(torch.sin(i),torch.cos(i))) for i in wgs_phases]

    temp_changes = []
    for i in range(1,len(temp_phases)):
        temp_changes.append(temp_phases[i] - temp_phases[i-1])

    wgs_changes = []
    for i in range(1,len(temp_phases)):
        wgs_changes.append(wgs_phases[i] - wgs_phases[i-1])

    ax = plt.subplot(2,1,1)
    ax.plot(temp_changes,label="Temporal")
    ax.plot(wgs_changes,label="WGS")
    ax.set_ylabel("Phase Change (rads)")
    ax.set_xlabel("Frame")

    ax.plot(torch.linspace(0,M,M),torch.ones(M)*T_in,label="T_in limit = pi/" + str(int(pi/T_in)),color="green")
    ax.plot(torch.linspace(0,M,M),torch.ones(M)*-1*T_in,color="green")
    # plt.yticks(torch.linspace(0,2*pi,10))
    ax.legend()

    ax = plt.subplot(2,1,2)
    ax.plot(temp_pressure,label="Temporal")
    ax.plot(wgs_pressure,label="WGS")
    ax.set_ylabel("Mean Pressure (Pa)")
    ax.set_xlabel("Frame")
    ax.set_ylim(0,9000)
    ax.legend()
    plt.show()
