from acoustools.Utilities import create_points, TOP_BOARD, BOTTOM_BOARD, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs, gspat, iterative_backpropagation, naive, gorkov_target
from acoustools.Levitator import LevitatorController
from acoustools.Mesh import load_scatterer
from acoustools.BEM import compute_E, get_cache_or_compute_H

import torch, time, pickle
from vedo import Mesh
from torch import Tensor
from types import FunctionType


def read_lcode(pth:str, ids:tuple[int]=(1000,), mesh:Mesh=None, thickness:float=0.001, BEM_path='../BEMMedia', 
               save_holo_name:str|None=None, wait_for_key_press:bool=False, C0_function:FunctionType|None = None, C0_params:dict={}, 
               extruder:Tensor|None = None, print_eval:bool=False, return_holos:bool=False, points_per_batch:int=1):
    '''
    Reads lcode and runs the commands on the levitator device \n
    :param pth: Path to lcode file
    :param ids: Ids for the levitator 
    :param mesh: Mesh to be printed
    :param thickness: The wall thickness of the object
    :param BEM_path: Path to BEM folder
    :param save_holo_name: Out to save holograms to, if any:
    :param wait_for_key_press: If true will wait for keypress after first hologram
    :param C0_function: Function to use when C0 command seen
    :param C0_params: Parameters for the function to use when C0 command seen
    :params print_eval: If True will print memory and time used
    :params return_holos: If True will save a return holograms
    '''

    iterations = 100
    board = TOP_BOARD
    A = None
    H = None
    solver = wgs
    delay = 0
    layer_z = 0
    cut_mesh = None
    in_function = None

    target_U = None
    target_P = None

    reflector = None

    current_points = []
    if extruder is None:
        extruder = create_points(1,1,0,-0.04, 0.04)
        
    extruder_text = str(extruder[:,0].item()) + ',' + str(extruder[:,1].item()) + ',' + str(extruder[:,2].item())
    last_L = 'L2'

    functions = {}

    start_from_focal_point = ['L0','L1','L2','L3']
    signature = ['Focal','Trap','Twin','Vortex']

    name_to_solver = {'wgs':wgs,'gspat':gspat, 'iterative_backpropagation':iterative_backpropagation,'naive':naive, 'gorkov_target':gorkov_target}

    lev = LevitatorController(ids=ids)

    t0 = time.time_ns()
    done_one_holo= False
    with open(pth,'r') as file:
        lines = file.read().rstrip().replace(';','').split('\n')
        # lines=lines[:-1]
        N_lines = len(lines)

        total_size = 0
        holograms = []
        for i,line in enumerate(lines):
            if print_eval: print(f"{i}/{len(lines)}", end='\r')
            line = line.rstrip()
            if  (line[0] != '#'): #ignore comments
                line = line.split('#')[0] #ignore comments
                groups = line.split(':')
                command = groups[0]

                if command.startswith('F'): #the command starts a Functions 
                    xs = functions[command]

                    lev.levitate(xs)
                    
        
                elif command in start_from_focal_point:
                    current_points.append(groups[1:])
                    if len(current_points) >= points_per_batch or (i + points_per_batch) >= points_per_batch:
                        x = L0(*current_points, iterations=iterations, board=board, A=A, solver=solver, mesh=cut_mesh,
                            BEM_path=BEM_path, H=H, U_target=target_U, reflector=reflector)
                        sig = signature[start_from_focal_point.index(command)]
                        x = add_lev_sig(x, board=board,mode=sig)
                        last_L = command

                        if in_function is not None:
                            functions[in_function].append(x)


                        total_size += x.element_size() * x.nelement()
                        if save_holo_name is not None or return_holos: 
                            holograms.append(x)
                        lev.levitate(x)

            

                        # layer_z = float(groups[1].split(',')[2])

                elif command == 'L4':
                    lev.turn_off()
                elif command == 'C0':
                    current_points_ext = current_points
                    current_points_ext.append(extruder_text)
                    x = L0(*current_points_ext, iterations=iterations, board=board, A=A, solver=solver, 
                           mesh=cut_mesh,BEM_path=BEM_path, H=H, reflector=reflector)
                    try:
                        sig = signature[start_from_focal_point.index(command)]
                    except ValueError as e:
                        sig = 'Twin'

                    x = add_lev_sig(x, board=board,mode=sig)
                                            
                    total_size += x.element_size() * x.nelement()
                    if save_holo_name is not None or return_holos: holograms.append(x.clone())
                    lev.levitate(x)

                    if wait_for_key_press :
                        input('Press enter to start...')
                    
                    if C0_function is not None:                    
                        C0_function(**C0_params)

                    else:
                        C0()
                elif command == 'C1':
                    C1()
                elif command == 'C2':
                    C2()
                elif command == 'C3':
                    time.sleep(float(groups[1])/1000)
                elif command == 'C4':
                    delay = float(groups[1])/1000
                elif command == 'C5':
                    solver= name_to_solver[groups[1].lower()]
                elif command == 'C6':
                    for group in groups:
                        if "I" in group:
                            it = group.split("I")[-1]
                            iterations = int(it)
                        if "U" in group:
                            U = group.split("U")[-1]
                            target_U = float(U)
                        if "P" in group:
                            P = group.split("P")[-1]
                            target_P = float(P)
                elif command == 'C7':
                    board = TRANSDUCERS
                elif command == 'C8':
                    board = TOP_BOARD
                elif command == 'C9':
                    board = BOTTOM_BOARD
                elif command == 'C10':
                    print('C10 Currently Disabled...')
                    # cut_mesh = cut_mesh_to_walls(mesh, layer_z=layer_z, wall_thickness=thickness)
                    # H = get_cache_or_compute_H(cut_mesh,board=board,path=BEM_path)
                elif command == 'C11':
                    frame_rate = float(groups[1])
                    lev.set_frame_rate(frame_rate)
                elif command == 'C12':
                    z = float(groups[1])
                    reflector = load_scatterer(BEM_path+'/flat-lam2.stl', dz=z)
                    H = get_cache_or_compute_H(reflector,board=board,path=BEM_path)
                elif command == 'function':
                    name = groups[1]
                    in_function = name
                    functions[name] = []
                elif command == 'end':
                    name = groups[1]
                    in_function = None
                elif command.startswith('O'):
                    pass
                else:
                    raise NotImplementedError(command)
                
                if delay > 0: time.sleep(delay)

    t1 = time.time_ns()
    if print_eval:
        print((t1-t0)/1e9,'seconds')
        print(total_size/1e6, 'MB')
    if save_holo_name is not None: pickle.dump(holograms, open(save_holo_name,'wb'))
    if return_holos: return holograms

def L0(*args, solver:FunctionType=wgs, iterations:int=50, board:Tensor=TOP_BOARD, A:Tensor=None, 
       mesh:Mesh=None, BEM_path:str='', H:Tensor=None, reflector:Mesh=None, U_target=None):
    '''
    @private
    '''
    batches = []
    for batch in args:
        ps = []
        for group in batch:
            group = [float(g) for g in group.split(',')]
            p = create_points(1,1,group[0], group[1], group[2])
            ps.append(p)
        points = torch.concatenate(ps, dim=2) 
        batches.append(points)
    points = torch.concatenate(batches, dim=0) 

    if (mesh is not None or reflector is not None) and A is None:
        if mesh is None and reflector is not None:
            A = compute_E(reflector, points=points, board=board, print_lines=False, path=BEM_path,H=H)
        else:
            A = compute_E(mesh, points=points, board=board, print_lines=False, path=BEM_path,H=H)
    
    if solver == wgs:
        x = wgs(points, iter=iterations,board=board, A=A )
    elif solver == gspat:
        x = gspat(points, board=board,A=A,iterations=iterations)
    elif solver == iterative_backpropagation:
        x = iterative_backpropagation(points, iterations=iterations, board=board, A=A)
    elif solver == naive:
        x = naive(points, board=board)
    elif solver == gorkov_target:
        x = gorkov_target(p,board=board, U_targets=U_target, reflector=reflector, path=BEM_path)
    else:
        raise NotImplementedError()
    

    return x

def C0(): #Dispense Droplets
    '''
    @private
    '''
    pass

def C1(): #Activate UV
    '''
    @private
    '''
    pass

def C2(): #Turn off UV
    '''
    @private
    '''
    pass

