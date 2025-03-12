'''

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
* `C12:<Extruder>;` Set a new extruder position
* `C13:<z>;` Use a reflector and set the position

* `O0;` End of droplet

* `function F<x>
...
end` define a function that can latter be called by name
'''

from torch import Tensor
from typing import Literal

def point_to_lcode(points:Tensor, sig_type:Literal['Focal', 'Trap', 'Vortex','Twin']='Focal') -> str:
    '''
    Converts AcousTools points to lcode string \n
    :param points: The points to export. Each batch will be a line in the reuslting lcode
    :param sig_type: The type of trap to create (defines L-command to use)
    :returns lcode: Lcode as a string
    '''


    N = points.shape[2]

    l_command = {'Focal':'L0','Trap':'L1','Twin':'L2','Vortex':'L3'}[sig_type.capitalize()]

    lcode = ''
    for batch in points: #Each batch should be a line
        lcode += '' + l_command
       
        for i in range(N):
                lcode += ':'
                p = batch[:,i]
                command = str(p[0].item()) + "," + str(p[1].item()) + ',' + str(p[2].item())

                lcode += command
        
        lcode += ';\n'

    return lcode.rstrip()
