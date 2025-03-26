from torch import Tensor
import torch
from typing import Literal


def is_batched_points(points:Tensor) -> bool:
    '''
    :param points: `Tensor` of points
    :return: `True` is points has a batched shape
    '''
    if len(points.shape)> 2 :
        return True
    else:
        return False
    


def permute_points(points: Tensor,index: int,axis:int=0) -> Tensor:
    '''
    Permutes axis of a tensor \n
    :param points: Tensor to permute
    :param index: Indexes describing order to perumte to 
    :param axis: Axis to permute. Default `0`
    :return: permuted points
    '''
    if axis == 0:
        return points[index,:,:,:]
    if axis == 1:
        return points[:,index,:,:]
    if axis == 2:
        return points[:,:,index,:]
    if axis == 3:
        return points[:,:,:,index]


def convert_to_complex(matrix: Tensor) -> Tensor:
    '''
    Comverts a real tensor of shape `B x M x N` to a complex tensor of shape `B x M/2 x N` 
    :param matrix: Matrix to convert
    :return: converted complex tensor
    '''
    # B x 1024 x N (real) -> B x N x 512 x 2 -> B x 512 x N (complex)
    matrix = torch.permute(matrix,(0,2,1))
    matrix = matrix.view((matrix.shape[0],matrix.shape[1],-1,2))
    matrix = torch.view_as_complex(matrix.contiguous())
    return torch.permute(matrix,(0,2,1))




def return_matrix(x,y,mat=None):
    '''
    @private
    Returns value of parameter `mat` - For compatibility with other functions
    '''
    return mat


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

def batch_list(iterable, batch=32):
    i = 0
    while i <= len(iterable):
        if i + batch <= len(iterable):
            yield iterable[i:i+batch]
        else:
            yield iterable[i:]
        i += batch