import torch
from acoustools.Utilities import propagate_abs, add_lev_sig, device, create_board, TRANSDUCERS, forward_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.colors as clrs
import matplotlib.cm as cm




def get_point_pos(A,B,C, points, res=(200,200),flip=True):
    '''
    converts point positions in 3D to pixel locations in the plane defined by ABC\\
    `A` Position of the top left corner of the image\\
    `B` Position of the top right corner of the image\\
    `C` Position of the bottom left corner of the image\\
    `res` Number of pixels as a tuple (X,Y). Default (200,200)\\
    `flip` Reverses X and Y directions. Default True
    '''
    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])

    ab_dir = AB!=0
    ac_dir = AC!=0

    step_x = AB / res[0]
    step_y = AC / res[1]

    if points.shape[2] > 1:
        points = torch.split(points.squeeze().T,1)
        points = [pt.squeeze() for pt in points]
    # print(points)

    pts_norm = []

    for pt in points:
        Apt =  torch.tensor([pt[0] - A[0], pt[1] - A[1], pt[2] - A[2]])
        px = Apt / step_x
        py = Apt / step_y
        pt_pos = torch.zeros((2))
        if not flip:
            pt_pos[0]= torch.round(px[ab_dir])
            pt_pos[1]=torch.round(py[ac_dir])
        else:
            pt_pos[1]= torch.round(px[ab_dir])
            pt_pos[0]=torch.round(py[ac_dir])
        
        pts_norm.append(pt_pos)

   

    return pts_norm

def Visualise_single(A,B,C,activation,colour_function=propagate_abs, colour_function_args={}, res=(200,200), flip=True):
    '''
    Visalises field generated from activation to the plane ABC
    `A` Position of the top left corner of the image\\
    `B` Position of the top right corner of the image\\
    `C` Position of the bottom left corner of the image\\
    `activation` The transducer activation to use\\
    `colour_function` Function to call at each position. Should return a value to colour the pixel at that position. Default `acoustools.Utilities.propagate_abs`\\
    `colour_function_args` The arguments to pass to `colour_function`\\
    `res` Number of pixels as a tuple (X,Y). Default (200,200)\\
    `flip` Reverses X and Y directions. Default True
    '''
    if len(activation.shape) < 3:
        activation = activation.unsqueeze(0)
    

    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])

    step_x = AB / res[0]
    step_y = AC / res[1]

    positions = torch.zeros((1,3,res[0]*res[1])).to(device)

    for i in range(0,res[0]):
        for j in range(res[1]):
            positions[:,:,i*res[0]+j] = A + step_x * i + step_y * j
    
    # print(positions.shape)
    # print(colour_function_args)
    field_val = colour_function(activation,positions,**colour_function_args)
    # print(field_val.shape)
    result = torch.reshape(field_val, res)

    if flip:
        result = torch.rot90(torch.fliplr(result))
    
    
    return result

def Visualise(A,B,C,activation,points=[],colour_functions=[propagate_abs], colour_function_args=None, 
              res=(200,200), cmaps=[], add_lines_functions=None, add_line_args=None,vmin=None,vmax=None, matricies = None, show=True,block=True, clr_labels=None):
    '''
    Visalises any numvber of fields generated from activation to the plane ABC and arranges them in a (1,N) grid
    `A` Position of the top left corner of the image\\
    `B` Position of the top right corner of the image\\
    `C` Position of the bottom left corner of the image\\
    `activation` The transducer activation to use\\
    `points` List of point positions to add crosses for each plot. Positions should be given in their position in 3D\\
    `colour_functions` List of function to call at each position for each plot. Should return a value to colour the pixel at that position. Default `acoustools.Utilities.propagate_abs`\\
    `colour_function_args` The arguments to pass to `colour_functions`\\
    `res` Number of pixels as a tuple (X,Y). Default (200,200)\\
    `cmaps` The cmaps to pass to plot\\
    `add_lines_functions` List of functions to extract lines and add to the image\\
    `add_line_args` List of parameters to add to `add_lines_functions`\\
    `vmin` Minimum value to use across all plots\\
    `vmax` MAximum value to use across all plots\\
    `matricies` precomputed matricies to plot\\
    `show` If True will call `plt.show(block=block)` else does not. Default True\\
    `block` Will be passed to `plot.show(block=block)`. Default True\\
    `clr_label`: Label for colourbar\\
    '''


    results = []
    lines = []
    if len(points) > 0:
        pts_pos = get_point_pos(A,B,C,points,res)
        # print(pts_pos)
        pts_pos_t = torch.stack(pts_pos).T


    if colour_function_args is None and colour_functions is not None:
        colour_function_args = [{}]*len(colour_functions)
    
    if colour_functions is not None:
        for i,colour_function in enumerate(colour_functions):
            result = Visualise_single(A,B,C,activation,colour_function, colour_function_args[i], res)
            results.append(result)
        
            if add_lines_functions is not None:
                if add_lines_functions[i] is not None:
                    lines.append(add_lines_functions[i](**add_line_args[i]))
                else:
                    lines.append(None)
    
    else:
        for i,mat in enumerate(matricies):
            result = mat
            print(result)
            results.append(result)
        
            if add_lines_functions is not None:
                if add_lines_functions[i] is not None:
                    lines.append(add_lines_functions[i](**add_line_args[i]))
                else:
                    lines.append(None)


    for i in range(len(results)):
        if len(cmaps) > 0:
            cmap = cmaps[i]
        else:
            cmap = 'hot'

        length = len(colour_functions) if colour_functions is not None else len(matricies)
        plt.subplot(1,length,i+1)
        im = results[i]
       
        v_min = vmin
        v_max = vmax
        
        if type(vmax) is list:
            v_max = vmax[i]
        
        if type(vmin) is list:
            v_min = vmin[i]
        
        if v_min is None:
            v_min = torch.min(im)
        if v_max is None:
            v_max = torch.max(im)
        

        # print(vmax,vmin)
        
        if clr_labels is None:
            clr_label = 'Pressure (Pa)'
        else:
            clr_label = clr_labels[i]
            
        plt.imshow(im.cpu().detach().numpy(),cmap=cmap,vmin=v_min,vmax=v_max)
        plt.colorbar(label=clr_label)

        if add_lines_functions is not None:
            AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]])
            AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]])
            # print(AB,AC)
            norm_x = AB
            norm_y = AC
            AB = AB[AB!=0] / res[0]
            AC = AC[AC!=0] / res[1]
            # AC = AC / torch.abs(AC)
            # print(AB,AC)
            if lines[i] is not None:
                for con in lines[i]:
                    xs = [con[0][0]/AB + res[0]/2, con[1][0]/AB + res[0]/2] #Convert real coordinates to pixels - number of steps in each direction
                    ys = [con[0][1]/AC + res[1]/2, con[1][1]/AC + res[1]/2] #Add res/2 as 0,0,0 in middle of real coordinates not corner of image
                    # print(xs,ys)
                    plt.plot(xs,ys,color = "blue")
        
        if len(points) >0:
            plt.scatter(pts_pos_t[1],pts_pos_t[0],marker="x")
    
    if show:
        plt.show(block=block)
    else:
        return plt

def force_quiver(points, U,V,norm, ylims=None, xlims=None,log=False,show=True,colour=None, reciprocal = False, block=True):

    B = points.shape[0]
    N = points.shape[2]
    
    # if len(points) > 0:
    #     pts_pos = get_point_pos(A,B,C,points,res)
    
    mask  = ~(torch.tensor(norm).to(bool))
    points = points[:,mask,:]
    # points=torch.reshape(points,(B,2,-1))
    

    xs = points[:,0,:].cpu().detach().numpy()[0]
    ys = points[:,1,:].cpu().detach().numpy()[0]


    if log:
        U = torch.sign(U) * torch.abs(torch.log(torch.abs(U)))   
        V = torch.sign(V) * torch.abs(torch.log(torch.abs(V))) 
    
    if reciprocal:
        U = 1/U
        V = 1/V
    

    plt.quiver(xs, ys, U.cpu().detach().numpy(),V.cpu().detach().numpy(),color = colour,linewidths=0.01)
    plt.axis('equal')


    if ylims is not None:
        plt.ylim(ylims[0],ylims[1])
    
    if xlims is not None:
        plt.xlim(xlims[0],xlims[1])
    
    if show:
        plt.show(block=block)
    


def force_quiver_3d(points, U,V,W, scale=1):
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(points[:,0,:].cpu().detach().numpy(), points[:,1,:].cpu().detach().numpy(), points[:,2,:].cpu().detach().numpy(), U.cpu().detach().numpy()* scale, V.cpu().detach().numpy()* scale, W.cpu().detach().numpy()* scale)
    plt.show()




def Visualise_mesh(mesh, colours=None, points=None, p_pressure=None,vmax=None,vmin=None, show=True, subplot=None, fig=None, buffer_x=0, buffer_y = 0, buffer_z = 0, equalise_axis=False, elev=-45, azim=45):


    xmin,xmax, ymin,ymax, zmin,zmax = mesh.bounds()
    
    if type(colours) is torch.Tensor:
        colours=colours.flatten()


    v = mesh.vertices
    f = torch.tensor(mesh.cells)

    if fig is None:
        fig = plt.figure()
    
    if subplot is None:
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.add_subplot(subplot,projection="3d")

    # norm = plt.Normalize(C.min(), C.max())
    # colors = plt.cm.viridis(norm(C))

    if vmin is None and colours is not None:
        vmin = torch.min(colours).item()
        if p_pressure is not None and p_pressure < vmin:
            vmin = p_pressure
    
    if vmax is None and colours is not None:
        vmax = torch.max(colours).item()
        if p_pressure is not None and p_pressure > vmax:
            vmax = p_pressure

    norm = clrs.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm, cmap=cm.hot)

    if points is not None:
        if p_pressure is not None:
            p_c = mapper.to_rgba(p_pressure.squeeze().cpu().detach())
        else:
            p_c = 'blue'
        points = points.cpu().detach()
        ax.scatter(points[:,0],points[:,1],points[:,2],color=p_c)

    if colours is not None:
        colour_mapped = []
        for c in colours:
            colour_mapped.append(mapper.to_rgba(c.cpu().detach()))
    else:
        colour_mapped=None

    pc = art3d.Poly3DCollection(v[f], edgecolor="black", linewidth=0.01, facecolors=colour_mapped)
    plt_3d = ax.add_collection(pc)

    scale = mesh.vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    if not equalise_axis:
        ax.set_xlim([xmin - buffer_x, xmax +  buffer_x])
        ax.set_ylim([ymin - buffer_y, ymax + buffer_y])
        ax.set_zlim([zmin - buffer_z, zmax + buffer_z])
    else:
        ax.set_aspect('equal')


    ax.view_init(elev=elev, azim=azim)


    if show:
        plt.show()
    else:
        return ax
    



def Visualise_line(A,B,x, F=None,points=None,steps = 1000, board=TRANSDUCERS, propagate_fun = propagate_abs, propagate_args={}, show=True):
    if points is None:
        AB = B-A
        step = AB / steps
        points = []
        for i in range(steps):
            p = A + i*step
            points.append(p.unsqueeze(0))
        
        points = torch.stack(points, 2).to(device)
    
    
    pressure = propagate_fun(activations=x,points=points, board=board,**propagate_args)
    if show:
        plt.plot(pressure.detach().cpu().flatten())
        plt.show()
    else:
        return pressure
       

def ABC(size, plane = 'xz'):
    if plane == 'xz':
        A = torch.tensor((-1,0, 1)) * size
        B = torch.tensor((1,0, 1))* size
        C = torch.tensor((-1,0, -1))* size
    
    if plane == 'yz':
        A = torch.tensor((0,-1, 1)) * size
        B = torch.tensor((0,1, 1))* size
        C = torch.tensor((0,-1, -1))* size
    

    

    return A, B, C