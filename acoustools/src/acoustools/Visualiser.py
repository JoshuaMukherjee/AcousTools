import torch
from acoustools.Utilities import propagate_abs, device, TRANSDUCERS, create_points
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
import matplotlib.colors as clrs
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import MultiCursor

import matplotlib.animation as animation
import matplotlib  as mpl

from torch import Tensor
from types import FunctionType
from typing import Literal
from vedo import Mesh





def Visualise(A:Tensor,B:Tensor,C:Tensor,activation:Tensor,points:list[Tensor]|Tensor=[],
              colour_functions:list[FunctionType]|None=[propagate_abs], colour_function_args:list[dict]|None=None, 
              res:tuple[int]=(200,200), cmaps:list[str]=[], add_lines_functions:list[FunctionType]|None=None, 
              add_line_args:list[dict]|None=None,vmin:int|list[int]|None=None,vmax:int|list[int]|None=None, 
              matricies:Tensor|list[Tensor]|None = None, show:bool=True,block:bool=True, clr_labels:list[str]|None=None, depth:int=2, link_ax:str|list|None='all',
              cursor:bool=False, arrangement:tuple|None = None, arangement:tuple|None = None, titles:list[str]|None=None, call_abs=False, norm_axes=None ) -> None:
    '''
    Visualises any number of fields generated from activation to the plane ABC and arranges them in a (1,N) grid \n
    :param A: Position of the top left corner of the image
    :param B: Position of the top right corner of the image
    :param C: Position of the bottom left corner of the image
    :param activation: The transducer activation to use
    :param points: List of point positions to add crosses for each plot. Positions should be given in their position in 3D
    :param colour_functions: List of function to call at each position for each plot. Should return a value to colour the pixel at that position. Default `acoustools.Utilities.propagate_abs`. \n
        - If colour_functions is `difference` or `ratio` then the result will be the difference or ratio between existing images. To control which images are used `colour_function_args` should have a parameter called `ids`.
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
    :param depth: Number of times to tile image
    :param link_ax: Axes to link colourbar of `'all'` to link all axes. To unlink all axis, pass one of ['none', False, None]
    :param cursor: If `True` will show cursor across plots
    :param arangement: Arangment of subplots 
    :param title: Titles for each subplot
    :param call_abs: if True will call torch.abs on the image 
    :param norm_axes: List of Axes to Normalise

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

    axs =[]
    results = []
    lines = []
    
    if arangement is None and arrangement is not None:
        arangement = arrangement


    if colour_function_args is None and colour_functions is not None:
        colour_function_args = [{}]*len(colour_functions)
    
    if type(activation) is not list :
        activation = [activation] *len(colour_functions)
    
    if type(A) != list:
        A = [A] * len(colour_functions)
    if type(B) != list:
        B = [B] * len(colour_functions)
    if type(C) != list:
        C = [C] * len(colour_functions)
    
    if colour_functions is not None:
        for i,colour_function in enumerate(colour_functions):
            if type(colour_function) is not str:
                if depth > 0:
                    result = Visualise_single_blocks(A[i],B[i],C[i],activation[i],colour_function, colour_function_args[i], res, depth=depth)
                else:
                    result = Visualise_single(A[i],B[i],C[i],activation[i],colour_function, colour_function_args[i], res)
                results.append(result)
            
                if add_lines_functions is not None:
                    if add_lines_functions[i] is not None:
                        lines.append(add_lines_functions[i](**add_line_args[i]))
                    else:
                        lines.append(None)
            else:  
                i1, i2 = colour_function_args[i]['ids'] if (i<len(colour_function_args) and 'ids' in colour_function_args[i]) else (0,1)
                img1 = results[i1]
                img2 = results[i2]

                if colour_function.lower() == 'difference' or colour_function.lower() == 'diff' or colour_function.lower() == '-':
                    image = (img2 - img1)
                if colour_function.lower() == 'ratio' or colour_function.lower() == 'rat' or colour_function.lower() == '/':
                    image = (img2 / img1)

                post_function= colour_function_args[i]['post_function'] if (i<len(colour_function_args) and 'post_function' in colour_function_args[i]) else None
                if post_function is not None:
                    image = post_function(image)

                results.append(image)

    
    else:
        for i,mat in enumerate(matricies):
            result = mat
            results.append(result)
        
            if add_lines_functions is not None:
                if add_lines_functions[i] is not None:
                    lines.append(add_lines_functions[i](**add_line_args[i]))
                else:
                    lines.append(None)

    v_min = vmin
    v_max = vmax
    
    if type(vmax) is list:
        v_max = vmax[i]
    
    if type(vmin) is list:
        v_min = vmin[i]
    
    norms = {}
    
    if link_ax == 'all' or link_ax == True:
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        for i in range(len(results)):
            norms[i] = norm
    elif link_ax == 'none' or link_ax is None or link_ax == False:
        for i in range(len(results)):
            norms[i] = None
    
    else:
        if type(link_ax[0]) == list or type(link_ax[0]) == tuple:
            for group in link_ax:
                norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
                for i in range(len(results)):
                    if i in group: 
                        norms[i] = norm 
                    elif i not in norms: 
                        norms[i] = None

        else:
            norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
            group = link_ax
            for i in range(len(results)):
                if i in group: 
                    norms[i] = norm
                elif i not in norms: 
                    norms[i] = None


    for i in range(len(results)):
        if len(cmaps) > 0:
            cmap = cmaps[i]
        else:
            cmap = 'hot'

        length = len(colour_functions) if colour_functions is not None else len(matricies)
        if arangement is None:
            arangement = (1,length)
        if i > 0:
            ax = plt.subplot(*arangement,i+1, sharex = ax, sharey=ax)
        else:
            ax = plt.subplot(*arangement,i+1)
        
        if titles is not None:
            t = titles[i]
            if t is not None:
                ax.set_title(t)

        axs.append(ax)
        im = results[i]
        if call_abs: im = torch.abs(im)
        

        if v_min is None:
            v_min = torch.min(im)
        if v_max is None:
            v_max = torch.max(im)
        

        # print(vmax,vmin)
        
        if clr_labels is None:
            clr_label = 'Pressure (Pa)'
        else:
            clr_label = clr_labels[i]
        
        
        if norm_axes is not None and i in norm_axes:
            im = im / torch.max(im)
        img = plt.imshow(im.cpu().detach().numpy(),cmap=cmap,norm=norms[i])
        plt.yticks([])
        plt.xticks([])

        if len(points) >0:
            
            if type(points) == list:
                points  = torch.concatenate(points,axis=0)
            pts_pos = get_point_pos(A[i],B[i],C[i],points.real,res)
            # print(pts_pos)
            pts_pos_t = torch.stack(pts_pos).T

            ax.scatter(pts_pos_t[1],pts_pos_t[0],marker="x")

        

        if im.shape[2] == 1:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            
            plt.colorbar(label=clr_label,cax=cax)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="0%", pad=0.05)
            cax.set_xticks([])
            cax.set_yticks([])


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
        
       
    
    fig = plt.gcf()
     
    c = MultiCursor(fig.canvas, axs, color='b',lw=0.5, horizOn=True, vertOn=True)

    multi_event_id = fig.canvas.mpl_connect('motion_notify_event', c.onmove)
        
    def press(event):
        nonlocal multi_event_id
        if event.key == 'z':
            if c.visible:
                fig.canvas.mpl_disconnect(multi_event_id)
                fig.canvas.draw_idle()
                c.visible = False
                c.active = True
            else: 
                multi_event_id = fig.canvas.mpl_connect('motion_notify_event', c.onmove)
                fig.canvas.draw_idle()
                for line in c.vlines + c.hlines:  
                    line.set_visible(True)
                c.visible = True
        if event.key == 'x' and c.visible:
            c.active = not c.active
            
    fig.canvas.mpl_connect('key_press_event', press)

    if not cursor:
        fig.canvas.mpl_disconnect(multi_event_id)
        fig.canvas.draw_idle()
        c.visible = False
        c.active = True


    
    if show:
        plt.show(block=block)
    else:
        return fig
    

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

def get_image_positions(A:Tensor,B:Tensor,C:Tensor, res:tuple[int]=(200,200)):
    '''
    Gets res[0] x res[1] points in the plane defined by ABC
    :param A: Position of the top left corner of the image
    :param B: Position of the top right corner of the image
    :param C: Position of the bottom left corner of the image
    :param res: Number of pixels as a tuple (X,Y). Default (200,200)
    :returnns positions: The positions of pixels
    '''
    AB = torch.tensor([B[0] - A[0], B[1] - A[1], B[2] - A[2]]).to(device)
    AC = torch.tensor([C[0] - A[0], C[1] - A[1], C[2] - A[2]]).to(device)

    step_x = AB / res[0]
    step_y = AC / res[1]

    positions = torch.zeros((1,3,res[0]*res[1])).to(device)

    for i in range(0,res[0]):
        for j in range(res[1]):
            positions[:,:,i*res[0]+j] = A + step_x * i + step_y * j
        
    return positions


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
    

    
    positions = get_image_positions(A,B,C,res)
   
    
    # print(positions.shape)
    # print(colour_function_args)
    field_val = colour_function(activations=activation,points=positions,**colour_function_args)
    if len(field_val.shape) < 3:
        field_val.unsqueeze_(2)
    results = []
    for i in range(field_val.shape[2]):
        result = torch.reshape(field_val[:,:,i], res)
        results.append(result)
    result = torch.stack(results,dim=2)

    if flip:
        result = torch.rot90(torch.fliplr(result))
    
    
    return result

def force_quiver(points: Tensor, U:Tensor,V:Tensor,norm:tuple[int]=(0,0,0), ylims:int|None=None, xlims:int|None=None,
                 log:bool=False,show:bool=True,colour:str|None=None, reciprocal:bool = False, block:bool=True, scale:float=1) -> None:
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
    :param scale: value to scale arrows by - note will pass 1/scale to matplotlib
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
    

    plt.quiver(xs, ys, U.cpu().detach().numpy(),V.cpu().detach().numpy(),color = colour,linewidths=0.01,scale=1/scale)
    plt.axis('equal')


    if ylims is not None:
        plt.ylim(ylims[0],ylims[1])
    
    if xlims is not None:
        plt.xlim(xlims[0],xlims[1])
    
    if show:
        plt.show(block=block)
    


def force_quiver_3d(points:Tensor, U:Tensor,V:Tensor,W:Tensor, scale:float=1, show:bool=True, ax:mpl.axes.Axes|None=None) ->None:
    '''
    Plot the force on a mesh in 3D
    :param points: The centre of the mesh faces
    :param U: Force in first axis
    :param V: Force in second axis
    :param W: Force in third axis
    :param scale: value to scale result by
    :param show: If True will call `plt.show()`
    :param ax: Axis to use 
    '''
    
    if ax is None: ax = plt.figure().add_subplot(projection='3d')
    ax.quiver(points[:,0,:].cpu().detach().numpy(), points[:,1,:].cpu().detach().numpy(), points[:,2,:].cpu().detach().numpy(), U.cpu().detach().numpy()* scale, V.cpu().detach().numpy()* scale, W.cpu().detach().numpy()* scale)
    
    if show: plt.show()
    else: return ax




def Visualise_mesh(mesh:Mesh, colours:Tensor|None=None, points:Tensor|None=None, p_pressure:Tensor|None=None,
                   vmax:int|None=None,vmin:int|None=None, show:bool=True, subplot:int|plt.Axes|None=None, fig:plt.Figure|None=None, 
                   buffer_x:int=0, buffer_y:int = 0, buffer_z:int = 0, equalise_axis:bool=False, elev:float=-45, azim:float=45, 
                   clamp:bool=False) ->None:
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
    :param clamp: if True will clamp values in colours to vmax and vmin
    '''

    xmin,xmax, ymin,ymax, zmin,zmax = mesh.bounds()
    
    if type(colours) is torch.Tensor:
        colours=colours.flatten()
    
    if clamp:
        colours = torch.clamp(colours,vmin,vmax)


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


    mappable = cm.ScalarMappable(cmap=cm.hot, norm=norm)
    mappable.set_array(colour_mapped)  # Link the data to the ScalarMappable

    # Add the color bar to the figure
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6)
    cbar.set_label('Face Value')


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
       

def ABC(size:float, plane:Literal['xz', 'yz', 'xy'] = 'xz', origin:Tensor|tuple=None) -> tuple[Tensor]:
    '''
    Get ABC values for visualisation
    * A top right corner
    * B top right corner
    * C bottom left corner
    :param size: The size of the window
    :param plane: Plane, one of 'xz' 'yz' 'xy'
    :param origin: The centre of the view window 
    :return: A,B,C 
    '''
    if origin is None:
        origin = torch.tensor((0,0,0), device=device)
    if type(origin) == tuple or type(origin) == list:
        origin = torch.tensor(origin).real
    
    origin = origin.squeeze().real

    
    if plane == 'xz':
        A = origin+torch.tensor((-1,0, 1), device=device) * size
        B = origin+torch.tensor((1,0, 1), device=device)* size
        C = origin+torch.tensor((-1,0, -1), device=device)* size
    
    if plane == 'yz':
        A = origin+torch.tensor((0,-1, 1), device=device) * size
        B = origin+torch.tensor((0,1, 1), device=device)* size
        C = origin+torch.tensor((0,-1, -1), device=device)* size
    
    if plane == 'xy':
        A = origin+torch.tensor((-1,1, 0), device=device) * size
        B = origin+torch.tensor((1, 1,0), device=device)* size
        C = origin+torch.tensor((-1, -1,0), device=device)* size
    
    

    return A.to(device), B.to(device), C.to(device)

def ABC_2_tiles(A:Tensor,B:Tensor,C:Tensor):
    '''
    Split ABC defined region into 4 tiles
    * A top right corner
    * B top right corner
    * C bottom left corner
    
    '''
    A1 = A
    B1 = A + (B-A)/2
    C1 = A+ (C-A)/2

    A2 = A+ (B-A)/2
    B2 = B
    C2 = A+ ((B-A)/2 + (C-A)/2)

    A3 = A+ (C-A)/2
    B3 = A + ((B-A)/2 + (C-A)/2)
    C3 = C

    A4 = A + ((B-A)/2 + (C-A)/2)
    B4 = A + (B-A)+(C-A)/2
    C4 = A + ((B-A)/2 + (C-A))

    return (A1,B1,C1), (A2,B2,C2), (A3,B3,C3), (A4,B4,C4)

def combine_tiles(t1:Tensor,t2:Tensor,t3:Tensor,t4:Tensor):
    '''
    Combines subimages into a larger image, used in `Visualise_single_blocks`
    :param t1: Top left image
    :param t2: Top right image
    :param t3: Bottom left image
    :param t4: Bottom right image
    '''
    top = torch.cat([t1,t2],dim=1)
    bottom = torch.cat([t3,t4],dim=1)

    return torch.cat([top,bottom],dim=0)

def Visualise_single_blocks(A:Tensor,B:Tensor,C:Tensor,activation:Tensor,
                     colour_function:FunctionType=propagate_abs, colour_function_args:dict={}, 
                     res:tuple[int]=(200,200), flip:bool=True, depth=2) -> Tensor:
    '''
    Visalises field generated from activation to the plane ABC in a slightly nicer memory efficient way by chunking into tiles
    :param A: Position of the top left corner of the image
    :param B: Position of the top right corner of the image
    :param C: Position of the bottom left corner of the image
    :param activation: The transducer activation to use
    :param colour_function: Function to call at each position. Should return a numeric value to colour the pixel at that position. Default `acoustools.Utilities.propagate_abs`
    :param colour_function_args: The arguments to pass to `colour_function`
    :param res: Number of pixels as a tuple (X,Y). Default (200,200)
    :param flip: Reverses X and Y directions. Default True
    :param depth: Number of times to chunk
    :return: Tensor of values of propagated field
    '''

    tiles = ABC_2_tiles(A,B,C)

    new_res = (int(res[0]/2), int(res[1]/2))

    ims = []


    for (nA,nB,nC) in tiles:
        if depth == 1:
            im = Visualise_single(nA,nB,nC,activation,colour_function=colour_function, colour_function_args=colour_function_args, res=new_res, flip=flip)
        else:
            im = Visualise_single_blocks(nA,nB,nC,activation,colour_function=colour_function, colour_function_args=colour_function_args, res=new_res, flip=flip, depth = depth-1)
        ims.append(im)
        # torch.cuda.empty_cache()

    im = combine_tiles(*ims)
    return im


def animate_lcode(pth, ax:mpl.axes.Axes|None=None, fig:plt.Figure=None, skip:int=1, show:bool=False, 
                  fname:str='', extruder:Tensor|None = None, xlims:tuple[float]|None=None, 
                  ylims:tuple[float]|None=None, zlims:tuple[float]|None=None, dpi:int=100, interval:int = 1, 
                  legend:bool=True, title:bool=True) -> None:
    '''
    Reads a .lcode file and produces a gif of the simulation of the result of that lcode\n
    :param pth: Path to the .lcode file
    :param ax: Axis to use, if None will create new
    :param fig: Figure to use, if None will create new
    :param skip: Number of instructions to skip per animation frame, default 1 (no skipping)
    :param show: If true will call plt.show()
    :param: fname: Name of file to save to
    :param extruder: If not None the position of the extruder to plot as Tensor
    :param xlims: Tuple of xlims, if None will use  (-0.12,0.12)
    :param ylims: Tuple of ylims, if None will use  (-0.12,0.12)
    :param zlims: Tuple of zlims, if None will use  (-0.12,0.12)
    :param dpi: dpi to use when saving gif
    :param inetrval: Time to wait between frames
    :param legend: If True will add figure legend
    :param title: If True will add figure title
    '''

    if fig is None: fig = plt.figure()
    if ax is None: ax = fig.add_subplot(projection='3d')
    

    point_commands = ['L0','L1','L2','L3']

    frames = []
    printed_points = [[],]

    functions = {}
    in_function = None

    

    with open(pth,'r') as file:
        lines = file.readlines()
        LINES = len(lines)
        for i,line in enumerate(lines):
            print(f"Line {i}/{LINES}", end='\r')
            line = line.replace(';','').rstrip()
            split = line.split(':')
            cmd = split[0]
            
            if cmd == 'function':
                name = split[1]
                functions[name] = []
                in_function = name
            
            elif cmd == 'end':
                name = split[1]
                in_function = None
            
            elif cmd.startswith('F'):
                frame_points = functions[cmd]
                for frame in frame_points:
                    frames.append(frame)
                    frame_printed_points = printed_points[-1].copy()
                    printed_points.append(frame_printed_points)



            elif cmd in point_commands:
                points = split[1:]
                ps = []
                for point in points:
                    ps.append(point.split(','))

                frame_points = [[float(p) for p in pt] for pt in ps]
                frames.append(frame_points)

                if in_function is not None:
                    functions[in_function].append(frame_points)
            
            frame_printed_points = printed_points[-1].copy()
            if cmd == 'C1':
                frame_printed_points.append(frames[-1].copy())
            
            printed_points.append(frame_printed_points)


    if extruder is not None:
        if type(extruder) is bool:
            extruder = create_points(1,1,0,-0.04, 0.04)
        ex_x = extruder[:,0].detach().cpu()
        ex_y = extruder[:,1].detach().cpu()
        ex_z = extruder[:,2].detach().cpu()


    FRAMES = int(len(frames) / skip)
    # FRAMES = 100

    if xlims is None:
        xlims = (-0.12,0.12)
    
    if ylims is None:
        ylims = (-0.12,0.12)
    
    if zlims is None:
        zlims = (-0.12,0.12)
    

    def traverse(index):
        index = index*skip
        ax.clear()
        print(f"Line {index/skip}/{FRAMES}", end='\r')
        for pt in frames[index]:
            ax.scatter(*pt, label='Trap')
        
        printed_xs = [i for i in [[p[0] for p in pt] for pt in printed_points[index]]]
        printed_ys = [i for i in [[p[1] for p in pt] for pt in printed_points[index]]]
        printed_zs = [i for i in [[p[2] for p in pt] for pt in printed_points[index]]]

        ax.scatter(printed_xs,printed_ys, printed_zs, label='Printed', edgecolor='black')

        ax.set_ylim(xlims)
        ax.set_xlim(ylims)
        ax.set_zlim(zlims)

        
        if extruder is not None: ax.scatter(ex_x, ex_y, ex_z, label='Extruder')

        if legend: ax.legend()
        if title: ax.set_title(f'Location: {index}')
        


    ani = animation.FuncAnimation(fig, traverse, frames=FRAMES, interval=interval)
    if show: plt.show()

    if fname == '':
        fname = 'Results.gif'
    ani.save(fname, writer='imagemagick', dpi = dpi)