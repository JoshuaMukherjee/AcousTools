from torch import Tensor
import torch
from typing import Literal, Iterable


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

def batch_list(iterable:Iterable, batch:int=32):
    '''
    Split an iterable into batch sized pieces\n
    :param iterable: The iterable to batch
    :param batch: The size to batch
    ```
    x = range(100)
    for b in batch_list(x):
        print(b)
    ```
    '''
    i = 0
    while i <= len(iterable):
        if i + batch <= len(iterable):
            yield iterable[i:i+batch]
        else:
            yield iterable[i:]
        i += batch

def get_rows_in(a_centres, b_centres, expand = True):
    '''
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