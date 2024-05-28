import torch
import itertools

def get_numeral(numeral, A, B, C):
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
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A + 0.5*AB + 0.1*AC)
    points.append(A + 0.5*AB + 0.9*AC)

    return points

def numeral_two(A,B,C):
    AB = B-A
    AC = C-A
    
    points = []

    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.9 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points

def numeral_three(A,B,C):
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
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.5*AB + 0.1 * AC)
    points.append(A+ 0.5*AB + 0.9 * AC)

    return points

def numeral_eight(A,B,C):
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
    AB = B-A
    AC = C-A

    points = []
    points.append(A+ 0.9*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.5 * AC)
    points.append(A+ 0.1*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.1 * AC)
    points.append(A+ 0.9*AB + 0.9 * AC)

    return points


def distance(p1, p2):
    return torch.sqrt(torch.sum((p2 - p1)**2))
    
def interpolate_points(p1, p2, n):
    vec = (p2 - p1) / n
    points = []
    for i in range(n):
        points.append(p1 + i * vec)

    return points

def interpolate_path(path, n):
    points = []
    total_dist = 0
    distances = []
    for p1, p2 in itertools.pairwise(path):
        d = distance(p1,p2)
        total_dist +=  d
        distances.append(d)

    for i,(p1, p2) in enumerate(itertools.pairwise(path)):
        d = distances[i]
        num = round((n * (d / total_dist )).item())
        points += interpolate_points(p1, p2, num)

    return points