'''
Export points to lcode 
Each line starts with a command (see below) followed by the arguments for that command. The command should be followed by a colon (:). 
Each line should end with a semi-colon (;), each argument is seperated by a comma (,) and groups or arguments can be seperated with a colon (:)

Commands
* `L0:<X> <Y> <Z>;` Create Focal Point at (X,Y,Z)
* `L1:<X> <Y> <Z>;` Create Trap Point at (X,Y,Z)
* `L2:<X> <Y> <Z>;` Create Twin Trap Point at (X,Y,Z)
* `L3:<X> <Y> <Z>;` Create Vortex Trap Point at (X,Y,Z)
* `L4;` Turn off Transducers

* `C0;`Dispense Droplet
* `C1;` Activate UV
* `C2;` Turn off UV
* `C3:<T>;` Delay for T ms
* `C4:<T>;` Set delay for T ms between all commands
* `C5:<Solver>;` Change to specific solver. Should be one of "IB", "WGS", "GSPAT", "NAIVE", "GORKOV_TARGET"
* `C6:I<I>:U<U>:P<P>;` Set parameters for the solver, I: Iterations. U:target Gorkov, P:Target Pressure. Note not all solvers support all options
* `C7;` Set to two board setup
* `C8;` Set to top board setup
* `C9;` Set to bottom board setup
* `C10;` Update BEM to use layer at last z position 
* `C11:<Frame-rate>;` Set the framerate of the levitator device
* `C12:<z>;` Use a reflector and set the position

* `O0;` End of droplet

* `function F<x>
...
end` define a function that can latter be called by name
'''

from torch import Tensor
from typing import Literal
from types import FunctionType

from acoustools.Solvers import wgs, iterative_backpropagation, gspat, naive, gorkov_target
from acoustools.Utilities import TOP_BOARD, BOTTOM_BOARD, TRANSDUCERS

def point_to_lcode(points:Tensor|list[Tensor], sig_type:Literal['Focal', 'Trap', 'Vortex','Twin']='Focal') -> str:
    '''
    Converts AcousTools points to lcode string \n
    :param points: The points to export. Each batch will be a line in the reuslting lcode
    :param sig_type: The type of trap to create (defines L-command to use)
    :returns lcode: Lcode as a string
    '''



    l_command = {'Focal':'L0','Trap':'L1','Twin':'L2','Vortex':'L3'}[sig_type.capitalize()]

    lcode = ''
    if type(points) == Tensor:
         points = [points,]
    for batches in points:    
        for batch in batches: #Each batch should be a line
            N = batch.shape[1]
            lcode += '' + l_command
        
            for i in range(N):
                    lcode += ':'
                    p = batch[:,i]
                    command = str(p[0].item()) + "," + str(p[1].item()) + ',' + str(p[2].item())

                    lcode += command
            
            lcode += ';\n'

    return lcode.rstrip()

def get_setup_commands(solver:FunctionType|None|str = None, I:int=200, U:float|None=None, P:int|None=None,
                       board:Tensor|None=None, frame_rate:int=200, flat_reflector_z:int|None=None) -> str:
    '''
    Creates setup commands
    :param Solver: String name or function of the solver to use
    :param I: Iterations
    :param U: Target Gorkov Value
    :param P: Target Pressure
    :param board: Board to use
    :param frame_rate: Levitator frame rate
    :param flat_reflector_z: if not None will use flat reflrctor at z position given\n
    :returns lcode:
    '''

    command = ''
    #Solver -> C5
    if solver is None:
        command += 'C5:WGS;\n'
    else:
        if type(solver) == FunctionType:
            solver = {iterative_backpropagation:"IB", wgs:"WGS", gspat:"GSPAT", naive:"NAIVE", gorkov_target:"GORKOV_TARGET"}[solver]
        command += f'C5:{solver};\n'

    #solver params
    command += f"C6:I{I}"
    if U is not None:
         command += f":U{U}"
    if P is not None:
         command += f":P{P}"
    command += ';\n'

    #Board 
    if board is None or ((board.shape == TRANSDUCERS.shape) and  (board == TRANSDUCERS).all()):
        command += 'C7;\n'
    
    elif (board.shape == TOP_BOARD.shape) and  (board == TOP_BOARD).all():
        command += 'C8;\n'
    
    elif (board.shape == BOTTOM_BOARD.shape) and (board == BOTTOM_BOARD).all():
        command += 'C9;\n'
    else:
         raise ValueError("Unknown board")
    
    #Frame Rate
    command += f"C11:{frame_rate};\n"

    #Reflector
    if flat_reflector_z is not None:
         command += f"C12:{flat_reflector_z};\n"

    return command.rstrip()


def export_to_lcode(fname, points:Tensor, sig_type:Literal['Focal', 'Trap', 'Vortex','Twin']='Focal',
                    solver:FunctionType|None|str = None, I:int=200, U:float|None=None, P:int|None=None,
                    board:Tensor|None=None, frame_rate:int=200, flat_reflector_z:int|None=None):
    
    setup_commands = get_setup_commands(solver, I, U, P,board, frame_rate, flat_reflector_z)
    point_command = point_to_lcode(points, sig_type)

    lcode = setup_commands + '\n' + point_command

    with open(fname,'w') as f:
         f.write(lcode)

         