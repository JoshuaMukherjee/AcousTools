import torch
import itertools
import math

from acoustools.Utilities import create_points, device

try:
    from svgpathtools import svg2paths, CubicBezier, Line
    svg_warning = False
except ImportError:
    svg_warning = True


from torch import Tensor
from types import FunctionType



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


def distance(p1:Tensor, p2:Tensor) -> float:
    '''
    Computes the euclidian distance between two points\n
    :param p1: First point
    :param p2: Second point
    :return: Distance
    '''
    return torch.sqrt(torch.sum((p2 - p1)**2)).real
    
def interpolate_points(p1:Tensor, p2:Tensor, n:int)-> list[Tensor]:
    '''
    Interpolates `n` points between `p1` and `p2`\n
    :param p1: First point
    :param p2: Second point
    :param n: number of points to interpolate
    :return: Path
    '''
    if n > 0:
        vec = (p2 - p1) / n
        points = []
        for i in range(n):
            points.append(p1 + i * vec)
    else:
        return p1

    return points


def total_distance(path: list[Tensor]):
    total_dist = 0
    distances = []
    for p1, p2 in itertools.pairwise(path):
        d = distance(p1,p2)
        total_dist +=  d
        distances.append(d)
    
    return total_dist, distances

def target_distance_to_n(total_dist, max_distance):
    n = total_dist / max_distance
    return math.ceil(n)

def interpolate_path(path: list[Tensor], n:int, return_distance:bool = False) -> list[Tensor]:
    '''
    Calls `interpolate_points on all adjacent pairs of points in path`\n
    :param n: TOTAL number of points to interpolate (will be split between pairs)
    :param return_distance: if `True` will also return total distance
    :return: Path and optionally total distance
    '''
    points = []
    total_dist, distances = total_distance(path)


    for i,(p1, p2) in enumerate(itertools.pairwise(path)):
        d = distances[i]
        num = round((n * (d / total_dist )).item())
        points += interpolate_points(p1, p2, num)

    if return_distance:
        return points, total_dist
    return points

def interpolate_path_to_distance(path: list[Tensor], max_diatance:float=0.001) -> list[Tensor]:
    '''
    Calls `interpolate_points on all adjacent pairs of points in path to make distance < max_distance` \n
    :param max_diatance: max_distance between adjacent points
    :return: Path and optionally total distance
    '''
    points = []

    for i,(p1, p2) in enumerate(itertools.pairwise(path)):
        total_dist, distances = total_distance([p1,p2])
        n = math.ceil(total_dist / max_diatance)
        points += interpolate_points(p1, p2, n)
    
    return points


def interpolate_arc(start:Tensor, end:Tensor|None=None, origin:Tensor=None, n:int=100, 
                    up:Tensor=torch.tensor([0,0,1.0]), anticlockwise:bool=False) -> list[Tensor]:
    
    '''
    Creates an arc between start and end with origin at origin\n
    :param start: Point defining start of the arc
    :param end: Point defining the end of the arc (this is not checked to lie on the arc - the arc will end at the same angle as arc)
    :param origin: Point defining origin of the arc
    :param n: number of points to interpolate along the arc. Default 100
    :param up: vector defining which way is 'up'. Default to positive z
    :param anticlickwise: If true will create anticlockwise arc. Otherwise will create clockwise arc
    :returns Points: List of points
    
    '''

    if origin is None:
        raise ValueError('Need to pass a value for origin')

    radius = torch.sqrt(torch.sum((start - origin)**2))

    start_vec = (start-origin)

    if end is not None:
        end_vec = (end-origin)
        cos = torch.dot(start_vec.squeeze(),end_vec.squeeze()) / (torch.linalg.vector_norm(start_vec.squeeze()) * torch.linalg.vector_norm(end_vec.squeeze()))
        angle = torch.acos(cos)
    else:
        end = start.clone() + 1e-10
        end_vec = (end-origin)
        angle = torch.tensor([3.14159 * 2])

    w = torch.cross(start_vec,end_vec,dim=1)
    clockwise = torch.dot(w.squeeze(),up.squeeze())<0

    u = start_vec 
    u/= torch.linalg.vector_norm(start_vec.squeeze())
    if  (w == 0).all():
        w += torch.ones_like(w) * 1e-10
    
    v = torch.cross(w,u,dim=1) 
    v /=  torch.linalg.vector_norm(v.squeeze())

    if clockwise == anticlockwise: #Should be false
        angle = 2*3.14159 - angle
        direction= -1
    else:
        direction = 1

    points = []
    for i in range(n):
            t = direction * ((angle) / n) * i
            p = radius * (torch.cos(t)*u + torch.sin(t)*v) + origin
            points.append(p)

    return points


def interpolate_bezier(start: Tensor, end:Tensor, offset_1:Tensor, offset_2:Tensor, n:int=100) -> list[Tensor]:

    '''
    Create cubic Bezier curve based on positions given \n
    :param start: Start position
    :param end: End position
    :param offset_1: offset from start to first control point
    :param offset_2: offset from start to second control point
    :param n: number of samples
    :returns points:
    '''

    #Make even sample?= distance

    P1 = start
    P2 = start + offset_1
    P3 = start + offset_2
    P4 = end

    points = []

    for i in range(n):
        t = i/n
        P5 = (1-t)*P1 + t*P2
        P6 = (1-t)*P2 + t*P3
        P7 = (1-t)*P3 + t*P4
        P8 = (1-t)*P5 + t*P6
        P9 = (1-t)*P6 + t*P7
        point = (1-t)*P8 + t*P9

        points.append(point)

    return points

def interpolate_bezier_velocity(start: Tensor, end:Tensor, offset_1:Tensor, offset_2:Tensor, n:int=100) -> list[Tensor]:
    '''
    Gets the velocity of a  cubic Bezier curve based on positions given \n
    :param start: Start position
    :param end: End position
    :param offset_1: offset from start to first control point
    :param offset_2: offset from start to second control point
    :param n: number of samples
    :returns points:
    '''

    P0 = start
    P1 = start + offset_1
    P2 = start + offset_2
    P3 = end

    points = []
    for i in range(n):
        t = i/n
        p = 3*(1-t)**2 *(P1-P0) + 6*(1-t)*t*(P2-P1) + 3*t**2 * (P3-P2)
        points.append(p)
    
    return points


def interpolate_bezier_acceleration(start: Tensor, end:Tensor, offset_1:Tensor, offset_2:Tensor, n:int=100) -> list[Tensor]:
    '''
    Gets the acceleration of a  cubic Bezier curve based on positions given \n
    :param start: Start position
    :param end: End position
    :param offset_1: offset from start to first control point
    :param offset_2: offset from start to second control point
    :param n: number of samples
    :returns points:
    '''
     
    P0 = start
    P1 = start + offset_1
    P2 = start + offset_2
    P3 = end

    points = []
    for i in range(n):
        t = i/n
        p = 6*(1-t)*(P2-2*P1+P0) + 6*t*(P3-2*P2+P1)
        points.append(p)
    
    return points


def svg_to_beziers(pth:str, flip_y:bool= False, n:int=20, dx:float=0, dy:float=0, scale_x:float = 1/10, scale_y:float = 1/10) -> tuple[list[Tensor]]:
    '''
    Converts a .SVG file containing bezier curves to a set of AcousTools bezier curves \n
    :param pth: String path to .svg file
    :param flip_y: If true flip the y axis
    :param n: Number of samples along bezier to return
    :param dx: change in x direction to apply
    :param dy: change in y direction to apply
    :param scale_x: scale in x direction to apply
    :param scale_y: scale in y direction to apply
    :returns (points, bezier): Points and the bezier curve as list of tuples. Bezier defined as (start, end, offset1, offset2) where offsets are from start
    '''
    if svg_warning:
        raise ImportError('Requires svgpathtools module `pip install svgpathtools`')
        

    paths, _ = svg2paths(pth)


    def ReIm_to_AcousTools_point(point, flip_y, dx, dy, scale_x, scale_y):
        if flip_y:
            y_mul = -1
        else:
            y_mul = 1

        point_AT = create_points(1,x=(point.real*scale_x) + dx, y=(y_mul*point.imag*scale_y)-dy,z=0)
        return point_AT

    points = []
    control_points = []
    i = -1

    for pth in paths:
        for bez in pth:
            if type(bez) == CubicBezier:
                i += 1

                start_RI = bez.start
                control_1_RI = bez.control1
                control_2_RI = bez.control2
                end_RI = bez.end

                start = ReIm_to_AcousTools_point(start_RI, flip_y, dx, dy, scale_x , scale_y)
                control1 = ReIm_to_AcousTools_point(control_1_RI, flip_y,dx, dy, scale_x , scale_y)
                control2 = ReIm_to_AcousTools_point(control_2_RI, flip_y,dx, dy, scale_x , scale_y)
                end = ReIm_to_AcousTools_point(end_RI, flip_y,dx, dy, scale_x , scale_y)

                control_points.append([start, end, control1-start, control2-start ])

                points += interpolate_bezier(start, end, control1-start, control2-start, n=n)

            elif type(bez) == Line:
                start_RI = bez.start
                end_RI = bez.end
                
                start = ReIm_to_AcousTools_point(start_RI, flip_y)
                end = ReIm_to_AcousTools_point(end_RI, flip_y)
                points += interpolate_path([start, end],n=n)
    

    xs = [p[:,0] for p in points]
    max_x = max(xs).clone()
    min_x = min(xs).clone()

    ys = [p[:,1] for p in points]
    max_y = max(ys).clone()
    min_y = min(ys).clone()
   

    return points, control_points


def bezier_to_C1(bezier:list[list[Tensor]], check_C0:bool=True, n:int=20) -> tuple[list[Tensor]]:
    '''
    Converts a bezier curve to be C1 continuous (https://en.wikipedia.org/wiki/Composite_B%C3%A9zier_curve#Smooth_joining)
    :param bezier: bezier curve to convert as (start, end, offset1, offset2) where offsets are from start 
    :param check_C0: If True will encure C0 continuity as well. Raises an error if violated
    :param n: number of samples
    :returns points,new_bezier: Points and new C1 bezier curve
    '''
    new_bezier = []
    new_bezier.append(bezier[0])

    for i,([P0, P3, c11, c12],[start_2,P6, c21, c22 ]) in enumerate(itertools.pairwise(bezier)):
        P1 = P0 + c11
        P2 = P0 + c12
        P5 = P3 + c22
 
        if check_C0: assert (P3 == start_2).all() #Assert we have C0 continuity

        P4_offset = (P3 - P2)

        new_bezier.append([P3, P6, P4_offset, c22])

    if (new_bezier[0][0] == new_bezier[-1][1]).all(): #C0 continuous at the last point -> Path is a loop
        [P0, P3, c11, c12] = new_bezier[-1]
        [start_2,P6, c21, c22 ] = new_bezier[0]

        P1 = P0 + c11
        P2 = P0 + c12
        P5 = P3 + c22
 
        if check_C0: assert (P3 == start_2).all() #Assert we have C0 continuity

        P4_offset = (P3 - P2)

        new_bezier[0] = [P3, P6, P4_offset, c22]





    points =[]
    for (P0, P3, c11, c12) in new_bezier:
        points += interpolate_bezier(P0,P3, c11, c12, n)
    
    
    return points,new_bezier

def close_bezier(bezier:list[list[Tensor]], n:int=20)  -> tuple[list[Tensor]]:
    '''
    Links the last point in a bezier to the start of it with a new bezier
    :param bezier: Bezier spline to close as list of (start, end, offset1, offset2) where offsets are from start 
    :param n: number of points to sample
    :returns points,bezier: points,bezier
    '''

    start = bezier[0]
    end = bezier[-1]

    new_b = [end[1], start[0],torch.zeros_like(start[0]),torch.zeros_like(start[0])]
    bezier.append(new_b)

    points =[]
    for (P0, P3, c11, c12) in bezier:
        points += interpolate_bezier(P0,P3, c11, c12, n)


    return points,bezier



def OptiSpline(bezier:list[list[Tensor]], target_points:list[Tensor], objective: FunctionType, 
               n:int=20, C1:bool=True, optimiser:torch.optim.Optimizer=torch.optim.Adam, 
               lr: float=0.01, objective_params:dict={},iters:int=200,log=True ):
    '''
    Optimiser for AcousTools bezier Splines \n
    :param bezier: Bezier spline as list of (start, end, offset1, offset2) where offsets are from start 
    :param target_points: Target points 
    :param objective: Objective function
    :param n: number of points to sample
    :param C1: If True will enforce C1 continuity
    :param optimiser: Optimiser to use - default Adam
    :param lr: learning rate to use - default 0.01
    :param objective_params: Objectives to pass to objective function
    :param iters: iterations to optimise for
    :param log: If true will print objective value at each step
    :returns bezier: Optimses curve

    '''
    params = []

    
    for bez in bezier:
        params.append(bez[2].requires_grad_())
        params.append(bez[3].requires_grad_())

    optim = optimiser(params,lr)

    target_points = torch.stack(target_points)
    
    for epoch in range(iters):

        optim.zero_grad()       

        loss = objective( bezier, target_points, n=n, **objective_params)
        if log: print(epoch, loss)

        loss.backward()
        optim.step()
        if C1: _,bezier=bezier_to_C1(bezier)




    return bezier