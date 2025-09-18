'''
Export Hologram to CSV
'''

from typing import Literal
from torch import Tensor
import torch

from acoustools.Utilities.Setup import device
from acoustools.Utilities.Utilities import get_convert_indexes


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
            
