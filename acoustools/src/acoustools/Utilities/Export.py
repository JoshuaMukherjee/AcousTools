
from typing import Literal
from torch import Tensor
import torch

from acoustools.Utilities import device

def get_convert_indexes(n:int=512, single_mode:Literal['bottom','top']='bottom') -> Tensor:
    '''
    Gets indexes to swap between transducer order for acoustools and OpenMPD for two boards\n
    Use: `row = row[:,FLIP_INDEXES]` and invert with `_,INVIDX = torch.sort(IDX)` 
    :param n: number of Transducers
    :param single_mode: When using only one board is that board a top or bottom baord. Default bottom
    :return: Indexes
    '''

    indexes = torch.arange(0,n)
    # # Flip top board
    # if single_mode.lower() == 'top':
    #     indexes[:256] = torch.flip(indexes[:256],dims=[0])
    # elif single_mode.lower() == 'bottom':
    #     indexes[:256] = torch.flatten(torch.flip(torch.reshape(indexes[:256],(16,-1)),dims=[1]))

    indexes[:256] = torch.flip(indexes[:256],dims=[0])
    
    if n > 256:
        indexes[256:] = torch.flatten(torch.flip(torch.reshape(indexes[256:],(16,-1)),dims=[1]))
    
    return indexes

def write_to_file(activations:Tensor,fname:str,num_frames:int, num_transducers:int=512, flip:bool=True) -> None:
    '''
    Writes each hologram in `activations` to the csv `fname` in order expected by OpenMPD \n
    :param activations: List of holograms
    :param fname: Name of file to write to, expected to end in `.csv`
    :param num_frames: Number of frames in `activations` 
    :param num_transducers: Number of transducers in the boards used. Default:512
    :param flip: If True uses `get_convert_indexes` to swap order of transducers to be the same as OpenMPD expects. Default: `True`
    '''
    output_f = open(fname,"w")
    output_f.write(str(num_frames)+","+str(num_transducers)+"\n")
    
    for row in activations:
        row = torch.angle(row).squeeze_()
        
        if flip:
            FLIP_INDEXES = get_convert_indexes()
            row = row[FLIP_INDEXES]
            

       
        for i,phase in enumerate(row):
                    output_f.write(str(phase.item()))
                    if i < num_transducers-1:
                        output_f.write(",")
                    else:
                        output_f.write("\n")

    output_f.close()

def get_rows_in(a_centres, b_centres, expand = True):
    '''
    @private
    Takes two tensors and returns a mask for `a_centres` where a value of true means that row exists in `b_centres` \\
    Asssumes in form 1x3xN -> returns mask over dim 1\\
    `a_centres` Tensor of points to check for inclusion in `b_centres` \\
    `b_centres` Tensor of points which may or maynot contain some number of points in `a_centres`\\
    `expand` if True returns mask as `1x3xN` if False returns mask as `1xN`. Default: True\\
    Returns mask for all rows in `a_centres` which are in `b_centres`
    '''

    M = a_centres.shape[2] #Number of total elements
    R = b_centres.shape[2] #Number of elements in b

    a_reshape = torch.unsqueeze(a_centres,3).expand(-1, -1, -1, R)
    b_reshape = torch.unsqueeze(b_centres,2).expand(-1, -1, M, -1)

    mask = b_reshape == a_reshape
    mask = mask.all(dim=1).any(dim=2)

    if expand:
        return mask.unsqueeze(1).expand(-1,3,-1)
    else:
        return mask

def read_phases_from_file(file: str, invert:bool=True, top_board:bool=False, ignore_first_line:bool=True):
    '''
    Gets phases from a csv file, expects a csv with each row being one geometry
    :param file: The file path to read from
    :param invert: Convert transducer order from OpenMPD -> Acoustools order. Default True
    :param top_board: if True assumes only the top board. Default False
    :param ignore_first_line: If true assumes header is the first line
    :return: phases
    '''
    phases_out = []
    line_one = True
    with open(file, "r") as f:
        for line in f.readlines():
            if ignore_first_line and line_one:
                line_one = False
                continue
            phases = line.rstrip().split(",")
            phases = [float(p) for p in phases]
            phases = torch.tensor(phases).to(device).unsqueeze_(1)
            phases = torch.exp(1j*phases)
            if invert:
                if not top_board:
                    IDX = get_convert_indexes()
                    _,INVIDX = torch.sort(IDX)
                    phases = phases[INVIDX]
                else:
                    for i in range(16):
                    #    print(torch.flipud(TOP_BOARD[i*16:(i+1)*16]))
                       phases[i*16:(i+1)*16] = torch.flipud(phases[i*16:(i+1)*16])
            phases_out.append(phases)
    phases_out = torch.stack(phases_out)
    return phases_out
            
