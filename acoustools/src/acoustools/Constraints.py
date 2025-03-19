import torch
from torch import Tensor


def constrain_amplitude(x:Tensor) -> Tensor:
    '''
    Constrains the amplitude of a hologram to be 1, χ(x) in AcousTools Paper\n
    :param x: Hologram
    :return: constrained hologram
    '''
    return x / torch.abs(x)

_abs = torch.abs
def constrain_field(field:Tensor, target:Tensor) -> Tensor:
    '''
    Constrains the amplitude of points in field to be the same as target\n
    :param field: propagated hologram-> points 
    :param target: complex number with target amplitude
    :return: constrained field
    '''
    field_amp = _abs(field)
    norm_field = field / field_amp
    target_field = target * norm_field 
    # target_field = torch.multiply(target,torch.divide(field,torch.abs(field)))  
    return target_field

def constrain_field_weighted(field:Tensor, target:Tensor, current:Tensor) -> tuple[Tensor, Tensor]:
    '''
    Constrains the amplitude of points in field to be the same as target with weighting\n
    :param field: propagated hologram-> points 
    :param target: complex number with target amplitude
    :param current: current amplitude of field
    :return: constrained weighted field
    '''

    current = target * current / torch.abs(field)


    current = current / torch.max(torch.abs(current),dim=1,keepdim=True).values
    field = constrain_field(field,current)
    return field, current