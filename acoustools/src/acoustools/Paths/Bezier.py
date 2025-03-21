import torch
from torch import Tensor

import itertools, math

try:
    from svgpathtools import svg2paths, Line
    from svgpathtools import CubicBezier as _CubicBezier
    svg_warning = False
except ImportError:
    svg_warning = True

from acoustools.Utilities.Points import create_points
from acoustools.Paths.Interpolate import interpolate_path, interpolate_points
from acoustools.Paths.Distances import distance

from acoustools.Paths.Curves import CubicBezier, Spline

def interpolate_bezier(bezier:CubicBezier, n:int=100) -> list[Tensor]:

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

    start,end,offset_1,offset_2 = bezier.get_data()

    
    if type(offset_1) == list: offset_1 = torch.tensor(offset_1).reshape((1,3,1))
    if type(offset_1) == int: offset_1 = torch.ones_like(start) * offset_1

    if type(offset_2) == list: offset_2 = torch.tensor(offset_2).reshape((1,3,1))
    if type(offset_2) == int: offset_2 = torch.ones_like(start) * offset_2


    P1 = start
    P2 = start + offset_1
    P3 = start + offset_2
    P4 = end

    points = []

    for i in range(n):
        t = i/n
        # THIS IS QUITE SLOW - REDUCE THIS TO ONE LINE - 
        P5 = (1-t)*P1 + t*P2
        P6 = (1-t)*P2 + t*P3
        P7 = (1-t)*P3 + t*P4
        P8 = (1-t)*P5 + t*P6
        P9 = (1-t)*P6 + t*P7
        point = (1-t)*P8 + t*P9

        points.append(point)

    return points

def interpolate_bezier_velocity(bezier:CubicBezier, n:int=100) -> list[Tensor]:
    '''
    Gets the velocity of a  cubic Bezier curve based on positions given \n
    :param start: Start position
    :param end: End position
    :param offset_1: offset from start to first control point
    :param offset_2: offset from start to second control point
    :param n: number of samples
    :returns points:
    '''

    start,end,offset_1,offset_2 = bezier.get_data()
    
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


def interpolate_bezier_acceleration(bezier:CubicBezier, n:int=100) -> list[Tensor]:
    '''
    Gets the acceleration of a  cubic Bezier curve based on positions given \n
    :param start: Start position
    :param end: End position
    :param offset_1: offset from start to first control point
    :param offset_2: offset from start to second control point
    :param n: number of samples
    :returns points:
    '''

    start,end,offset_1,offset_2 = bezier.get_data()
     
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


def svg_to_beziers(pth:str, flip_y:bool= False, n:int=20, dx:float=0, dy:float=0, scale_x:float = 1/10, scale_y:float = 1/10) -> tuple[list[Tensor], Spline]:
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
            if type(bez) == _CubicBezier:
                i += 1

                start_RI = bez.start
                control_1_RI = bez.control1
                control_2_RI = bez.control2
                end_RI = bez.end

                start = ReIm_to_AcousTools_point(start_RI, flip_y, dx, dy, scale_x , scale_y)
                control1 = ReIm_to_AcousTools_point(control_1_RI, flip_y,dx, dy, scale_x , scale_y)
                control2 = ReIm_to_AcousTools_point(control_2_RI, flip_y,dx, dy, scale_x , scale_y)
                end = ReIm_to_AcousTools_point(end_RI, flip_y,dx, dy, scale_x , scale_y)
                
                b = CubicBezier(start, end, control1-start, control2-start)
                control_points.append(b)

                points += interpolate_bezier(b, n=n)

            elif type(bez) == Line:
                start_RI = bez.start
                end_RI = bez.end
                
                start = ReIm_to_AcousTools_point(start_RI, flip_y)
                end = ReIm_to_AcousTools_point(end_RI, flip_y)
                points += interpolate_path([start, end],n=n)
    

    # xs = [p[:,0] for p in points]
    # max_x = max(xs).clone()
    # min_x = min(xs).clone()

    # ys = [p[:,1] for p in points]
    # max_y = max(ys).clone()
    # min_y = min(ys).clone()
   

    return points, Spline(control_points)

def bezier_to_C1(spline:Spline, check_C0:bool=True, n:int=20, get_points=True) -> tuple[list[Tensor]]:
    '''
    Converts a spline of beziers to be C1 continuous (https://en.wikipedia.org/wiki/Composite_B%C3%A9zier_curve#Smooth_joining)
    :param spline: spline of curves to convert  
    :param check_C0: If True will encure C0 continuity as well. Raises an error if violated
    :param n: number of samples
    :returns points,new_bezier: Points and new C1 spline curve
    '''
    # new_bezier = CubicBezier(None,None,None,None)
    # new_bezier.start = bezier[0]
    # new_bezier.append(bezier[0])

    new_spline = Spline()
    new_spline.add_curve(spline[0])

    for i,(b1,b2) in enumerate(itertools.pairwise(spline)):
        P0, P3, c11, c12 = b1.get_data()
        start_2,P6, c21, c22 = b2.get_data()
        P1 = P0 + c11
        P2 = P0 + c12
        P5 = P3 + c22
 
        # if check_C0: assert (P3 == start_2).all() #Assert we have C0 continuity

        P4_offset = (P3 - P2)

        new_spline.add_curve(CubicBezier(P3, P6, P4_offset, c22))

    if (new_spline[0][0] == new_spline[-1][1]).all(): #C0 continuous at the last point -> Path is a loop
        [P0, P3, c11, c12] = new_spline[-1].get_data()
        [start_2,P6, c21, c22 ] = new_spline[0].get_data()

        P1 = P0 + c11
        P2 = P0 + c12
        P5 = P3 + c22
 
        # if check_C0: assert (P3 == start_2).all() #Assert we have C0 continuity

        P4_offset = (P3 - P2)

        new_spline[0] = CubicBezier(P3, P6, P4_offset, c22)


    if get_points:
        points =[]
        for bez in new_spline:
            points += interpolate_bezier(bez, n)
    
    
        return points,new_spline

    return new_spline

def close_bezier(spline:Spline, n:int=20)  -> tuple[list[Tensor]]:
    '''
    Links the last point in a bezier to the start of it with a new bezier \n
    :param bezier: Bezier spline to close as list of (start, end, offset1, offset2) where offsets are from start 
    :param n: number of points to sample
    :returns points,bezier: points,bezier
    '''

    start = spline[0]
    end = spline[-1]

    new_b = [end[1], start[0],torch.zeros_like(start[0]),torch.zeros_like(start[0])]
    spline.add_curve(CubicBezier(*new_b))

    if n != 0:
        points =[]
        for bez in spline:
            points += interpolate_bezier(bez, n)
        return points,spline
    return spline

def bezier_to_distance(bezier:CubicBezier, max_distance:float=0.001, start_n=20):
    '''
    Samples bezier to have at most max_distance between points. \n
    :param bezier:`acoustools.Paths.Curves.Bezier` object
    :param max_distance: maximum straight line distance between points
    :param start_n: number of points to start
    :returns points:
    '''
    bezier_points = interpolate_bezier(bezier,n=start_n)

    points = []
    
    for i,(p1,p2) in enumerate(itertools.pairwise(bezier_points)):
        
        n = int(torch.ceil(torch.max(distance(p1, p2) / max_distance)).item())
        points += interpolate_points(p1,p2, n)
    
    return points

def create_bezier_circle(N=10, origin=(0,0,0), radius=0.01, plane='xy'):
    angle = 3.14*2 / N
    origin = create_points(1,1,origin[0], origin[1],origin[2])
    beziers = []
    for i in range(N):
        pos_1 = radius * math.sin(angle*i) 
        pos_2 = radius * math.cos(angle*i)

        if plane == 'xy':
            start = create_points(1,1,pos_1,pos_2,0) + origin
        elif plane == 'xz':
            start = create_points(1,1,pos_1,0,pos_2) + origin
        elif plane == 'yz':
            start = create_points(1,1,0,pos_1,pos_2) + origin
        else:
            raise ValueError("Plane not valid. Must be xy, xz or yz")
        
        pos_3 = radius * math.sin(angle*(i+1)) 
        pos_4 = radius * math.cos(angle*(i+1))

        if plane == 'xy':
            end = create_points(1,1,pos_3,pos_4,0) + origin
        elif plane == 'xz':
            end = create_points(1,1,pos_3,0,pos_4) + origin
        elif plane == 'yz':
            end = create_points(1,1,0,pos_3,pos_4) + origin
        else:
            raise ValueError("Plane not valid. Must be xy, xz or yz")

        offset_1 = create_points(1,1,0,0,0)

        bez = CubicBezier(start,end,offset_1, offset_1.clone())
        beziers.append(bez)
    spline = Spline(beziers)

    return spline


def connect_ends(spline:Spline):
    '''
    Sets the end of the last curve in spline to be the start of the first
    '''
    start = spline[0]
    end = spline[-1]

    end.end = start.start