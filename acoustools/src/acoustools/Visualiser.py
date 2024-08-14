import torch
from acoustools.Utilities import propagate_abs, add_lev_sig, device, create_board, TRANSDUCERS, forward_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.colors as clrs
import matplotlib.cm as cm

from torch import Tensor
from types import FunctionType
from typing import Literal
from vedo import Mesh


def Visualise(A:Tensor,B:Tensor,C:Tensor,activation:Tensor,points:list[Tensor]|Tensor=[],
              colour_functions:list[FunctionType]|None=[propagate_abs], colour_function_args:list[dict]|None=None, 
              res:tuple[int]=(200,200), cmaps:list[str]=[], add_lines_functions:list[FunctionType]|None=None, 
              add_line_args:list[dict]|None=None,vmin:int|list[int]|None=None,vmax:int|list[int]|None=None, 
              matricies:Tensor|list[Tensor]|None = None, show:bool=True,block:bool=True, clr_labels:list[str]|None=None) -> None:
    '''
    Visualises any number of fields generated from activation to the plane ABC and arranges them in a (1,N) grid \n
    :param A: Position of the top left corner of the image
    :param B: Position of the top right corner of the image
    :param C: Position of the bottom left corner of the image
    :param activation: The transducer activation to use
    :param points: List of point positions to add crosses for each plot. Positions should be given in their position in 3D
    :param colour_functions: List of function to call at each position for each plot. Should return a value to colour the pixel at that position. Default `acoustools.Utilities.propagate_abs`
    :param colour_function_args: The arguments to pass to `colour_functions`
    :param res: Number of pixels as a tuple (X,Y). Default (200,200)
    :param cmaps: The cmaps to pass to plot
    :param add_lines_functions: List of functions to extract lines and add to the image
    :param add_line_args: List of parameters to add to `add_lines_functions`
    :param vmin: Minimum value to use across all plots
    :param vmax: MAximum value to use across all plots
    :param matricies: precomputed matricies to plot
    :param show: If True will call `plt.show(block=block)` else does not. Default True
    :param block: Will be passed to `plot.show(block=block)`. Default True
    :param clr_label: Label for colourbar

    ```Python
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs
    from acoustools.Visualiser import Visualise

    import torch

    p = create_points(1,1,x=0,y=0,z=0)
    x = wgs(p)
    x = add_lev_sig(x)

    A = torch.tensor((-0.09,0, 0.09))
    B = torch.tensor((0.09,0, 0.09))
    C = torch.tensor((-0.09,0, -0.09))
    normal = (0,1,0)
    origin = (0,0,0)

    Visualise(A,B,C, x, points=p)
    ```
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
    

def get_point_pos(A:Tensor,B:Tensor,C:Tensor, points:Tensor, res:tuple[int]=(200,200),flip:bool=True) -> list[int]:
    '''
    converts point positions in 3D to pixel locations in the plane defined by ABC\n
    :param A: Position of the top left corner of the image
    :param B: Position of the top right corner of the image
    :param C: Position of the bottom left corner of the image
    :param res: Number of pixels as a tuple (X,Y). Default (200,200)
    :param flip: Reverses X and Y directions. Default True
    :return: List of point positions
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

def Visualise_single(A:Tensor,B:Tensor,C:Tensor,activation:Tensor,
                     colour_function:FunctionType=propagate_abs, colour_function_args:dict={}, 
                     res:tuple[int]=(200,200), flip:bool=True) -> Tensor:
    '''
    Visalises field generated from activation to the plane ABC
    :param A: Position of the top left corner of the image
    :param B: Position of the top right corner of the image
    :param C: Position of the bottom left corner of the image
    :param activation: The transducer activation to use
    :param colour_function: Function to call at each position. Should return a numeric value to colour the pixel at that position. Default `acoustools.Utilities.propagate_abs`
    :param colour_function_args: The arguments to pass to `colour_function`
    :param res: Number of pixels as a tuple (X,Y). Default (200,200)
    :param flip: Reverses X and Y directions. Default True
    :return: Tensor of values of propagated field
    '''
    if len(activation.shape) < 3:
        activation = activation.unsqueeze(0)
    

    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]]).to(device)
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]]).to(device)

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

def force_quiver(points: Tensor, U:Tensor,V:Tensor,norm:Tensor, ylims:int|None=None, xlims:int|None=None,
                 log:bool=False,show:bool=True,colour:str|None=None, reciprocal:bool = False, block:bool=True) -> None:
    '''
    Plot the force on a mesh as a quiver plot\n
    :param points: The centre of the mesh faces
    :param U: Force in first axis
    :param V: Force in second axis
    :param norm:
    :param ylims: limit of y axis
    :param zlims: limit of x axis
    :param log: if `True` take the log of the values before plotting
    :param show: if `True` call `plt.show()`
    :param colour: colour of arrows
    :param reciprocal: if `True` plot reciprocal of values
    :param block: passed into `plt.show`
    '''

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
    


def force_quiver_3d(points:Tensor, U:Tensor,V:Tensor,W:Tensor, scale:float=1) ->None:
    '''
    Plot the force on a mesh in 3D
    :param points: The centre of the mesh faces
    :param U: Force in first axis
    :param V: Force in second axis
    :param W: Force in third axis
    :param scale: value to scale result by
    '''
    
    ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(points[:,0,:].cpu().detach().numpy(), points[:,1,:].cpu().detach().numpy(), points[:,2,:].cpu().detach().numpy(), U.cpu().detach().numpy()* scale, V.cpu().detach().numpy()* scale, W.cpu().detach().numpy()* scale)
    plt.show()




def Visualise_mesh(mesh:Mesh, colours:Tensor|None=None, points:Tensor|None=None, p_pressure:Tensor|None=None,
                   vmax:int|None=None,vmin:int|None=None, show:bool=True, subplot:int|plt.Axes|None=None, fig:plt.Figure|None=None, 
                   buffer_x:int=0, buffer_y:int = 0, buffer_z:int = 0, equalise_axis:bool=False, elev:float=-45, azim:float=45) ->None:
    '''
    Plot a mesh in 3D and colour the mesh faces
    :param mesh: Mesh to plot
    :param colours: Colours for each face
    :param points: Positions of points to also plot
    :param p_pressure: Values to colour points with
    :param vmax: Maximum colour to plot
    :param vmin: Minimum colour to plot
    :param show: If `True` call `plot.show()`
    :param subplot: Optionally use existing subplot
    :param fig: Optionally use existing fig
    :param buffer_x: Amount of whitesapce to add in x direction
    :param buffer_y: Amount of whitesapce to add in y direction
    :param buffer_z: Amount of whitesapce to add in z direction
    :param equalise_axis: If `True` call `ax.set_aspect('equal')`
    :param elev: elevation angle
    :param azim: azimuth angle
    '''

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
    



def Visualise_line(A:Tensor,B:Tensor,x:Tensor, F:Tensor|None=None,points:Tensor|None=None,steps:int = 1000, 
                   board:Tensor|None=None, propagate_fun:FunctionType = propagate_abs, propagate_args:dict={}, show:bool=True) -> None:
    '''
    Plot the field across a line from A->B\n
    :param A: Start of line
    :param B: End of line
    :param x: Hologram
    :param F: Optionally, propagation matrix
    :param points: Optionally, pass the points on line AB instead of computing them
    :param steps: Number of points along line
    :param board: Transducers to use
    :param propagate_fun: Function to use to propagate hologram
    :propagate_args: arguments for `propagate_fun`
    :show: If `True` call `plt.show()`
    '''
    if board is None:
        board = TRANSDUCERS
    
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
       

def ABC(size:int, plane:Literal['xz', 'yz', 'xy'] = 'xz') -> tuple[Tensor]:
    '''
    Get ABC values for visualisation
    * A top right corner
    * B bottom right corner
    * C top left corner
    :param size: The size of the window
    :param plane: Plane, one of 'xz' 'yz' 'xy'
    :return: A,B,C 
    '''
    if plane == 'xz':
        A = torch.tensor((-1,0, 1)) * size
        B = torch.tensor((1,0, 1))* size
        C = torch.tensor((-1,0, -1))* size
    
    if plane == 'yz':
        A = torch.tensor((0,-1, 1)) * size
        B = torch.tensor((0,1, 1))* size
        C = torch.tensor((0,-1, -1))* size
    
    if plane == 'xy':
        A = torch.tensor((-1,1, 0)) * size
        B = torch.tensor((1, 1,0))* size
        C = torch.tensor((-1, -1,0))* size
    
    

    return A, B, C