'''
Translates gcode file to lcode file 

'''
# NEED TO ADD FRAME RATE CONTROL


from acoustools.Utilities import create_points
from acoustools.Paths import interpolate_points, distance, interpolate_arc, interpolate_bezier, bezier_to_distance

from torch import Tensor
import torch
from typing import Literal


def gcode_to_lcode(fname:str, output_name:str|None=None, output_dir:str|None=None, log:bool=True, log_name:str|None=None, log_dir:str=None,
                    divider:float = 1000, relative:bool = False, 
                   max_stepsize:float=0.001, extruder:Tensor|None = None, pre_print_command:str = '', 
                   post_print_command:str = '', print_lines:bool=False, pre_commands:str= '', post_commands:str='', 
                   use_BEM:bool = False, sig_type:str='Trap', travel_type:Literal["hypot","legsXY","legsZ","bezier"]='hypot',
                   add_optimisation_commands:bool=True, via:Tensor|None=None, use_functions:bool=False):
    '''
    Converts a .gcode file to a .lcode file \n
    ```Python
    from acoustools.Fabrication.Translater import gcode_to_lcode


    pth = 'acoustools/tests/data/gcode/rectangle.gcode'

    pre_cmd = 'C0;\\n'
    post_cmd = 'C1;\\nC3:10;\\nC2;\\n'
    gcode_to_lcode(pth, pre_print_command=pre_cmd, post_print_command=post_cmd)

    ```
    :param fname: The file name of the gcode file
    :param output_name: The filename for the lcode file, if None will use the gcode file name with .gcode replaced with .lcode
    :param output_dir: output directory of the lcode file, if None will use the same as the gcode file
    :param log: If True will save log files 
    :param log_name: Name for the log file, if None will will use the gcode file name with .gcode replaced with .txt and the name with '_log' appended
    :param log_dir: Directory for log file, if None will use same as the gcode file
    :param divider: Value to divide dx,dy,dz by - useful to change units
    :param relative: If true will change relative to last position, else will be absolute
    :param max_stepsize: Maximum stepsize allowed, default 1mm
    :param extruder: Extruder location, if None will use (0,0.10, 0)
    :param pre_print_command: commands to put before the block resulting from each G01, G02 and G03 command
    :param post_print_command: commands to put before the block resulting from each G01, G02 and G03 command
    :param print_lines: If true will print which line is being processed
    :param pre_commands: commands to put at the top of the file
    :param post_commands: commands to put at the end of the file
    :param add_optimisation_commands: If True will add commands to help `acoustools.Fabrication.Optimsation` functions work correctly
    :param via: A point to move all paths through before going to the end point
    :param use_functions: If true will collapse common code into functions - needs `via` to be specified too
    :param use_BEM: If True will use scattering for the mesh while it is built
    :param sig_type: Defined signature to use for traps
    :param travel_type: Way to move the particle

    '''


    name = fname.replace('.gcode','')
    parts = name.split('/')
    name = parts[-1]
    path = '/'.join(parts[:-1])

    if extruder is None:
        extruder = create_points(1,1,0,-0.04, 0.04)

    if output_name is None:
        output_name = name

    if output_dir is None:
        output_dir = path
    
    if log_dir is None:
        log_dir = output_dir
    
    if log_name is None:
        log_name = name 

    if use_functions:
        functions = {}
    else:
        functions = None
    
    
    output_file = open(output_dir+'/'+output_name+'.lcode','w')
    output_file.write(pre_commands)

    if log: log_file = open(log_dir+'/'+log_name+'_log.txt','w')

    head_position = create_points(1,1,0,0,0)
    E_val = 0
    


    with open(fname) as file:
        lines = file.readlines()
        Nl = len(lines)
        for i,line in enumerate(lines):
            if print_lines and i % 10 == 0: print(f'Computing, {i}/{Nl}', end='\r')
            line = line.rstrip()
            line_split = line.split()
            if len(line_split) == 0 or line.startswith(';'):
                if log: log_file.write(f'Line {i+1}, Ignoring line {line})\n')
                continue
            code = line_split[0]
            args = line_split[1:]
            command = ''
            
            E_val = get_E_val(*args, E_val=E_val)

            if (code == 'G00' or code == 'G0') or ((code == 'G01' or code == 'G1') and E_val == 0): #Non-fabricating move
                head_z = head_position[:,2].item()
                command, head_position = convert_G00(*args, head_position=head_position, divider=divider, relative=relative)
                if use_BEM and head_z != head_position[:,2]:
                    command += 'C10;\n'
                
                if log: log_file.write(f'Line {i+1}, G00 Command: Virtual head updated to {head_position[:,0].item()}, {head_position[:,1].item()}, {head_position[:,2].item()} ({line}), E value set to {E_val} \n')
            elif code == 'G01' or code == 'G1': #Fabricating move
                command, head_position, N = convert_G01(*args, head_position=head_position, extruder=extruder, divider=divider, relative=relative, 
                                                        max_stepsize=max_stepsize,pre_print_command=pre_print_command, 
                                                        post_print_command=post_print_command, sig=sig_type, travel_type=travel_type, via=via,
                                                        functions=functions)
                
                if log: log_file.write(f'Line {i+1}, G01 Command: Line printed to {head_position[:,0].item()}, {head_position[:,1].item()}, {head_position[:,2].item()} in {N} steps ({line}), E value set to {E_val} \n')
            
            elif code == 'G02' or code == 'G2': #Fabricating move
                command, head_position, N = convert_G02_G03(*args, head_position=head_position, extruder=extruder, divider=divider, relative=relative, 
                                                            max_stepsize=max_stepsize, anticlockwise=False,pre_print_command=pre_print_command, 
                                                            post_print_command=post_print_command, sig=sig_type, 
                                                            travel_type=travel_type, add_optimisation_commands=add_optimisation_commands , via=via,
                                                            functions=functions)

                if log: log_file.write(f'Line {i+1}, G02 Command: Circle printed to {head_position[:,0].item()}, {head_position[:,1].item()}, {head_position[:,2].item()} in {N} steps ({line}) \n')
            
            elif code == 'G03' or code == 'G3': #Fabricating arc
                command, head_position, N = convert_G02_G03(*args, head_position=head_position, extruder=extruder, divider=divider, relative=relative, 
                                                            max_stepsize=max_stepsize, anticlockwise=True,pre_print_command=pre_print_command, 
                                                            post_print_command=post_print_command, sig=sig_type, 
                                                            travel_type=travel_type , add_optimisation_commands=add_optimisation_commands , via=via,
                                                            functions=functions)

                if log: log_file.write(f'Line {i+1}, G03 Command: Circle printed to {head_position[:,0].item()}, {head_position[:,1].item()}, {head_position[:,2].item()} in {N} steps ({line}) \n')
            elif code == 'G04' or code == 'G4': #Dwell!!
                pass 
            elif code.startswith(';'):
                if log: log_file.write(f'Line {i+1}, Ignoring Comment ({line})\n')

            else: #Ignore everything else
                if log: log_file.write(f'Line {i+1}, Ignoring code {code} ({line})\n')
            
            output_file.write(command)
        
    
    output_file.write(post_commands)
    output_file.close()
    if print_lines: print()
    if log: log_file.close()

def get_E_val(*args:str, E_val:float):
    E = E_val
    for arg in args:
        if arg.startswith('E'):
            E = float(arg[1:])
    return E

def parse_xyz(*args:str):
    '''
    Takes the args from a gcode line and ginds the XYZ arguments \n
    :param args: list of arguments from gcode (all of the line but the command)
    '''
    x,y,z = None,None,None #Need to check if any of these need to not be changed
    for arg in args:
        arg = arg.lower()
        if 'x' in arg:
            x = float(arg.replace('x',''))
        
        if 'y' in arg:
            y = float(arg.replace('y',''))
        
        if 'z' in arg:
            z = float(arg.replace('z',''))

    return x,y,z

def update_head(head_position: Tensor, dx:float, dy:float, dz:float, divider:float, relative:bool):
    '''
    Updates the head (or other tensor) based on dx,dy,dz \n
    :param head_position: Start position
    :param dx: change (or new value) for x axis
    :param dy: change (or new value) for y axis
    :param dz: change (or new value) for z axis
    :param divider: Value to divide dx,dy,dz by - useful to change units
    :param relative: If true will change relative to last position, else will set head_position
    '''
    if relative:
        if dx is not None: head_position[:,0] += dx/divider 
        if dy is not None: head_position[:,1] += dy/divider
        if dz is not None: head_position[:,2] += dz/divider
    else:
        if dx is not None: head_position[:,0] = dx/divider 
        if dy is not None: head_position[:,1] = dy/divider
        if dz is not None: head_position[:,2] = dz/divider

def extruder_to_point(points:list[Tensor], extruder:Tensor, max_stepsize:float=0.001, travel_type:str='hypot', via:Tensor|None=None ) -> list[Tensor]:
    '''
    Will create a path from the extruder to each point in a shape \n
    :param points: Points in shape
    :param extruder: Extruder location
    :param max_stepsize: Maximum stepsize allowed, default 1mm
    :param travel_type: Type of movement to use, hypot: will use the shortest path, legsXY: will move in the XY plane then Z plane, legsZ: will move in the Z plane then XY plane, bezier, Will use a quadratic bezier in 3D
    :param via: A point to move all paths through before going to the end point
    :returns all points in path: 
    '''
    
    #No path planning -> Talk to Pengyuan? 

    all_points = []
    for p in points:
        if via is not None:
            all_points += start_to_end(extruder, via, max_stepsize, travel_type) #can be replaced by a 'function' call
            all_points += start_to_end(via, p, max_stepsize, travel_type)
        else:
            all_points += start_to_end(extruder, p, max_stepsize, travel_type)
    
    return all_points

def start_to_end(end:Tensor, start:Tensor, max_stepsize:float=0.001, travel_type:str='hypot'):
    '''
    Will create a path from the a start position to an end \n
    :param points: start point location
    :param end: end point location
    :param max_stepsize: Maximum stepsize allowed, default 1mm
    :returns all points in path: 
    '''

    points = []
    if travel_type == 'legsXY': #Move in XY plane then move in Z
    
        mid_point = create_points(1,1,x=start[0].item(), y=start[1].item(), z=end[:,2].item())
        d = distance(start, mid_point)
        N  = int(torch.ceil(torch.max(d / max_stepsize)).item())
        points += interpolate_points(end, mid_point, N)

        d = distance(mid_point, start)
        N  = int(torch.ceil(torch.max(d / max_stepsize)).item())
        points += interpolate_points(mid_point, start, N)
    
    elif travel_type == 'legsZ': #Move in XY plane then move in Z
    
        mid_point = create_points(1,1,x=end[:,0].item(), y=end[:,1].item(), z=start[2].item())
        

        d = distance(mid_point, start)
        N  = int(torch.ceil(torch.max(d / max_stepsize)).item())
        points += interpolate_points(end, mid_point, N)

        d = distance(start, mid_point)
        N  = int(torch.ceil(torch.max(d / max_stepsize)).item())
        points += interpolate_points(mid_point, start, N)

    elif travel_type == 'bezier': #Move along Bezier curve - paramatarised? 
        mid_point = create_points(1,1,x=start[0].item(), y=start[1].item(), z=end[:,2].item())
        offset_2 = mid_point - end

        bezier = [end,  start, [0,0,0], offset_2]
        
        points += bezier_to_distance(bezier)

    else: #default is hypot
        d = distance(start, end)
        N  = int(torch.ceil(torch.max(d / max_stepsize)).item())
        points += interpolate_points(end, start, N)
    
    return points



def points_to_lcode_trap(points:list[Tensor], sig:str='Trap', function_name:str='') -> tuple[str,Tensor]:
    '''
    Converts a set of points to a number of L1 commands (Traps) \n
    :param points: The point locations
    :param sig: The signature name to use
    :param function_name: The name of the function these commands make up, if any
    :returns command, head_position: The commands as string and the final head position
    '''
    command = ''
    if function_name != '':
        command += f"function:{function_name}: \n"
    for point in points:
        N = point.shape[2]
        sig_num = {'Focal':'0','Trap':"1",'Twin':'2','Vortex':'3'}[sig]
        command += f"L{sig_num}:"
        for i in range(N):
            command += f'{point[:,0].item()},{point[:,1].item()},{point[:,2].item()}'
            if i+1 < N:
                command += ':'
        command += ';\n'
    
        head_position = point
    
    if function_name != '':
        command += f"end:{function_name}; \n"

    return command, head_position

def convert_G00(*args:str, head_position:Tensor, divider:float = 1000, relative:bool=False) -> tuple[str, Tensor]:
    '''
    Comverts G00 commands to virtual head movements \n
    :param args: Arguments to G00 command
    :param head_position: strt position
    :param divider: Value to divide dx,dy,dz by - useful to change units
    :param relative: If true will change relative to last position, else will set head_position
    :returns '', head_position: Returns an empty command and the new head position
    '''
    dx, dy, dz = parse_xyz(*args)

    update_head(head_position, dx, dy, dz, divider, relative)

    return '', head_position

def convert_G01(*args:str, head_position:Tensor, extruder:Tensor, divider:float = 1000, 
                relative:bool=False, max_stepsize:bool=0.001, pre_print_command:str = '', post_print_command:str = '', 
                sig:str='Trap', travel_type:str='hypot', add_optimisation_commands:bool=True, via:Tensor|None=None,functions:dict|None=None) -> tuple[str, Tensor]:
    '''
    Comverts G00 commands to line of points \n
    :param args: Arguments to G00 command
    :param head_position: strt position
    :param extruder: Extruder location
    :param divider: Value to divide dx,dy,dz by - useful to change units
    :param relative: If true will change relative to last position, else will set head_position
    :param max_stepsize: Maximum stepsize allowed, default 1mm
    :param pre_print_command: commands to put before generated commands
    :param post_print_command: commands to put after generated commands
    :param sig: Signature to use 
    :param add_optimisation_commands: If True will add commands to help `acoustools.Fabrication.Optimsation` functions work correctly
    :param via: A point to move all paths through before going to the end point
    :returns command, head_position: Returns the commands and the new head position

    '''
    dx, dy, dz = parse_xyz(*args)

    end_position = head_position.clone()

    update_head(end_position, dx, dy, dz, divider, relative)

    N = int(torch.ceil(torch.max(distance(head_position, end_position) / max_stepsize)).item())
    if N > 0:
        print_points = interpolate_points(head_position, end_position,N)

        command, head_position = print_points_to_commands(print_points, extruder=extruder, max_stepsize=max_stepsize, 
                                                      pre_print_command=pre_print_command, post_print_command=post_print_command,
                                                      sig=sig, travel_type=travel_type, add_optimisation_commands=add_optimisation_commands, via=via,
                                                     functions=functions)
    else:
        command = ''
    return command, end_position, N

def convert_G02_G03(*args, head_position:Tensor, extruder:Tensor, divider:float = 1000, 
                    relative:bool=False, max_stepsize:float=0.001, anticlockwise:bool = False, 
                    pre_print_command:str = '', post_print_command:str = '', sig:str='Trap', 
                    travel_type:str='hypot', add_optimisation_commands:bool=True, via:Tensor|None=None, functions:dict|None=None)-> tuple[str, Tensor]:
    '''
    Comverts G02 and G03 commands to arc of points \n
    :param args: Arguments to G00 command
    :param head_position: strt position
    :param extruder: Extruder location
    :param divider: Value to divide dx,dy,dz by - useful to change units
    :param relative: If true will change relative to last position, else will set head_position
    :param max_stepsize: Maximum stepsize allowed, default 1mm
    :param anticlockwise: If true will arc anticlockwise, otherwise clockwise
    :param pre_print_command: commands to put before generated commands
    :param post_print_command: commands to put after generated commands
    :param sig: Signature to use 
    :param add_optimisation_commands: If True will add commands to help `acoustools.Fabrication.Optimsation` functions work correctly
    :param via: A point to move all paths through before going to the end point
    :returns command, head_position: Returns the commands and the new head position
    '''

    dx, dy, dz = parse_xyz(*args)

    end_position = head_position.clone()
    origin = head_position.clone()

    update_head(end_position, dx, dy, dz, divider, relative)

    for arg in args:
        arg = arg.lower()
        if 'i' in arg:
            I = float(arg.replace('i',''))
        
        if 'j' in arg:
            J = float(arg.replace('j',''))
    
    update_head(origin, I, J, 0, divider, relative)
    radius = distance(head_position, origin)

    start_vec = (head_position-origin)
    end_vec = (end_position-origin)
    cos = torch.dot(start_vec.squeeze(),end_vec.squeeze()) / (torch.linalg.vector_norm(start_vec.squeeze()) * torch.linalg.vector_norm(end_vec.squeeze()))
    
    angle = torch.acos(cos)

    d = angle * radius
    N = int(torch.ceil(torch.max( d / max_stepsize)).item())
    print_points = interpolate_arc(head_position, end_position, origin,n=N,anticlockwise=anticlockwise)
    
    command, head_position = print_points_to_commands(print_points, extruder=extruder, max_stepsize=max_stepsize, 
                                                      pre_print_command=pre_print_command, post_print_command=post_print_command,
                                                      sig=sig, travel_type=travel_type, add_optimisation_commands=add_optimisation_commands, via=via, 
                                                      functions=functions)

    return command, end_position, N

def print_points_to_commands(print_points, extruder:Tensor, max_stepsize:float=0.001, 
                    pre_print_command:str = '', post_print_command:str = '', sig:str='Trap', 
                    travel_type:str='hypot', add_optimisation_commands:bool=True, via:Tensor|None=None, functions:dict|None={}):
    '''
    @private
    '''
    command = ''
    for point in print_points:
        if via is not None and functions is not None:
            '''Check if we can use a function - if we have seen the same (extruder, travel_type, max_stepsize, via) before'''
            extruder_x = extruder[:,0].item()
            extruder_y = extruder[:,1].item()
            extruder_z = extruder[:,2].item()

            via_x = extruder[:,0].item()
            via_y = extruder[:,1].item()
            via_z = extruder[:,2].item()

            idx = str(extruder_x) + '_' + str(extruder_y)+ '_' + str(extruder_z)+ '_' + travel_type+ '_' + \
                    str(max_stepsize) + '_'+ str(via_x)+ '_' + str(via_y) + '_'+ str(via_z)
            
            if idx not in functions: #This is the first time we've seen this so create it

                if 'names' not in functions:
                    functions['names'] = 0
                
                name = functions['names'] + 1 
                functions['names'] = name
                
                pt = extruder_to_point(via, extruder, travel_type=travel_type, max_stepsize=max_stepsize)
                cmd, head_position =  points_to_lcode_trap(pt, sig=sig,function_name=f"F{name}")
                command += pre_print_command
                command += cmd
                
                functions[idx] = (f"F{name};\n", head_position)
                

            else:
                '''Just use precomputed path'''
                cmd, head_position = functions[idx]
                command += pre_print_command
                command += cmd
            
            pt = extruder_to_point(point, via, travel_type=travel_type, max_stepsize=max_stepsize)
            cmd, head_position =  points_to_lcode_trap(pt, sig=sig)
            command += cmd
            command += post_print_command
            if add_optimisation_commands: command += 'O0;\n'
            
            
        else:


            pt = extruder_to_point(point, extruder, travel_type=travel_type, max_stepsize=max_stepsize, via=via)
            cmd, head_position =  points_to_lcode_trap(pt, sig=sig)
            command += pre_print_command
            command += cmd
            command += post_print_command
            if add_optimisation_commands: command += 'O0;\n'
    
    return command, head_position
