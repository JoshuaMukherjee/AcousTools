

import torch

from torch import Tensor
from vedo import Mesh

def board_name(board:Tensor) -> str:
    '''
    Returns the name for a board, TOP and/or BOTTOM, used in cache system
    :param board: The board to use
    :return: name of board as `<'TOP'><'BOTTOM'><M>` for `M` transducers in the boards 
    '''
    M = board.shape[0]

    top = "TOP" if 1 in torch.sign(board[:,2]) else ""
    bottom = "BOTTOM" if -1 in torch.sign(board[:,2]) else ""
    return top+bottom+str(M)

def scatterer_file_name(scatterer:Mesh) ->str:
    '''
    Get a unique name to describe a scatterer position, calls `str(scatterer.coordinates)`
    ONLY USE TO SET FILENAME, USE `scatterer.filename` TO GET
    :param scatterer: The Mesh to use
    :return: Scatterer name
    
    '''

    f_name = str(list(scatterer.coordinates)) + str(scatterer.cell_normals)
    return f_name