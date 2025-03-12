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
