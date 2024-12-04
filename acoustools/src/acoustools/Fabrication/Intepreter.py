from acoustools.Utilities import create_points, TOP_BOARD, BOTTOM_BOARD, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs, gspat, iterative_backpropagation, naive
from acoustools.Levitator import LevitatorController
from acoustools.Mesh import cut_mesh_to_walls
from acoustools.BEM import compute_E, get_cache_or_compute_H

import torch, time, pickle
from vedo import Mesh
import vedo
from torch import Tensor
from types import FunctionType

def read_lcode(pth:str, ids:tuple[int]=(1000,), mesh:Mesh=None, thickness:float=0.001, BEM_path='../BEMMedia', save_holo_name:str|None=None):
    '''
    Reads lcode and runs the commands on the levitator device \n
    :param pth: Path to lcode file
    :param ids: Ids for the levitator 
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

    functions = {}

    start_from_focal_point = ['L0','L1','L2','L3']
    signature = ['Focal','Trap','Twin','Vortex']

    name_to_solver = {'wgs':wgs,'gspat':gspat, 'iterative_backpropagation':iterative_backpropagation,'naive':naive}

    lev = LevitatorController(ids=ids)

    t0 = time.time_ns()
    with open(pth,'r') as file:
        lines = file.read().rstrip().replace(';','').split('\n')
        lines=lines[:-1]

        total_size = 0
        holograms = []
        for i,line in enumerate(lines):
            print(f"{i}/{len(lines)}", end='\r')
            line = line.rstrip()
            if  (line[0] != '#'): #ignore comments
                line = line.split('#')[0] #ignore comments
                groups = line.split(':')
                command = groups[0]

                if command.startswith('F'):
                    xs = functions[command]
                    lev.levitate(xs)

                elif command in start_from_focal_point:

                    x = L0(*groups[1:], iterations=iterations, board=board, A=A, solver=solver, mesh=cut_mesh,BEM_path=BEM_path, H=H)
                    sig = signature[start_from_focal_point.index(command)]
                    x = add_lev_sig(x, board=board,mode=sig)

                    if in_function is not None:
                        functions[in_function].append(x)


                    total_size += x.element_size() * x.nelement()
                    if save_holo_name is not None: holograms.append(x)
                    lev.levitate(x)

                    layer_z = float(groups[1].split(',')[2])

                elif command == 'L4':
                    lev.turn_off()
                elif command == 'C0':
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
                    solver= name_to_solver[groups[1]]
                elif command == 'C6':
                    iterations = int(groups[1])
                elif command == 'C7':
                    board = TRANSDUCERS
                elif command == 'C8':
                    board = TOP_BOARD
                elif command == 'C9':
                    board = BOTTOM_BOARD
                elif command == 'C10':
                    cut_mesh = cut_mesh_to_walls(mesh, layer_z=layer_z, wall_thickness=thickness)
                    H = get_cache_or_compute_H(cut_mesh,board=board,path=BEM_path)
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
                
                time.sleep(delay)

    t1 = time.time_ns()
    print((t1-t0)/1e9,'seconds')
    print(total_size/1e6, 'MB')
    if save_holo_name is not None: pickle.dump(holograms, open(save_holo_name,'wb'))

def L0(*args, solver:FunctionType=wgs, iterations:int=50, board:Tensor=TOP_BOARD, A:Tensor=None, mesh:Mesh=None, BEM_path:str='', H:Tensor=None):
    '''
    @private
    '''
    ps = []
    for group in args:
        group = [float(g) for g in group.split(',')]
        p = create_points(1,1,group[0], group[1], group[2])
        ps.append(p)
    points = torch.concatenate(ps, dim=2) 

    if mesh is not None and A is None:
        A = compute_E(mesh, points=points, board=board, print_lines=False, path=BEM_path,H=H)
    
    if solver == wgs:
        x = wgs(points, iter=iterations,board=board, A=A )
    elif solver == gspat:
        x = gspat(points, board=board,A=A,iterations=iterations)
    elif solver == iterative_backpropagation:
        x = iterative_backpropagation(points, iterations=iterations, board=board, A=A)
    elif solver == naive:
        x = naive(points, board=board)
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

