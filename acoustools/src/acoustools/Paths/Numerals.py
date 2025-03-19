import torch
from torch import Tensor



def get_numeral(numeral:int|str, A:Tensor, B:Tensor, C:Tensor) -> list[Tensor]:
    '''
    Get positions to draw a given numeral \n
    :param numeral: The number to generate a path for [1,9] 
    :param A: top left corner of the box to draw in
    :param B: bottom left corner of the box to draw in
    :param C: top right corner of the box to draw in
    :return: path

    '''
    if type(A) is not torch.Tensor or type(B) is not torch.Tensor or type(C) is not torch.Tensor:
        A = torch.Tensor(A)
        B = torch.Tensor(B)
        C = torch.Tensor(C)
    if int(numeral) == 1:
        return numeral_one(A,B,C)
    if int(numeral) == 2:
        return numeral_two(A,B,C)
    if int(numeral) == 3:
        return numeral_three(A,B,C)
    if int(numeral) == 4:
        return numeral_four(A,B,C)
    if int(numeral) == 5:
        return numeral_five(A,B,C)
    if int(numeral) == 6:
        return numeral_six(A,B,C)
    if int(numeral) == 7:
        return numeral_seven(A,B,C)
    if int(numeral) == 8:
        return numeral_eight(A,B,C)
    if int(numeral) == 9:
        return numeral_nine(A,B,C)

def numeral_one(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A + 0.5*AB + 0.1*AC)
    points.append(A + 0.5*AB + 0.9*AC)

    return points

def numeral_two(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points

def numeral_three(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.3 * AC)
    points.append(A+ 0.3*AB + 0.5 * AC)
    points.append(A+ 0.9*AB + 0.7 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)

    return points

def numeral_four(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points

def numeral_five(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.3 * AC)
    points.append(A+ 0.9*AB + 0.6 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)

    return points

def numeral_six(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)

    return points

def numeral_seven(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.5*AB + 0.1 * AC)
    points.append(A+ 0.5*AB + 0.9 * AC)

    return points

def numeral_eight(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.5*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)
    points.append(A+ 0.5*AB + 0.5 * AC)

    return points

def numeral_nine(A,B,C):
    '''
    @private
    '''
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points
