from torch import Tensor
import torch

from acoustools.Utilities import forward_model, device



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

def swap_output_to_activations(out_mat,points):
    '''
    @private
    '''
    acts = None
    for i,out in enumerate(out_mat):
        out = out.T.contiguous()
        pressures =  torch.view_as_complex(out)
        A = forward_model(points[i]).to(device)
        if acts == None:
            acts =  A.T @ pressures
        else:
            acts = torch.stack((acts,A.T @ pressures),0)
    return acts

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

