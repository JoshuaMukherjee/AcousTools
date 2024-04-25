import torch


def constrain_amplitude(x):
    '''
    Constrains the amplitude of a hologram to be 1, χ(x) in AcousTools\\
    `x`: Hologram\\
    Returns constrained hologram
    '''
    return x / torch.abs(x)

def constrain_field(field, target):
    '''
    Constrains the amplitude of points in field to be the same as target\\
    `field` propagated hologram-> points \\
    `target` complex number with target amplitude\\
    Returns constrained field
    '''
    target_field = torch.multiply(target,torch.divide(field,torch.abs(field)))  
    return target_field

def constrain_field_weighted(field, target, current):
    '''
    Constrains the amplitude of points in field to be the same as target with weighting\\
    `field` propagated hologram-> points \\
    `target` complex number with target amplitude\\
    `current` current amplitude of field
    Returns constrained weighted field
    '''
    current = target * current / torch.abs(field)
    current = current / torch.max(torch.abs(current),dim=1).values
    field = constrain_field(field,current)
    return field, current