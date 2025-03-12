from acoustools.Intepreter import read_lcode
import pickle
from acoustools.Visualiser import Visualise, ABC
from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points
from acoustools.BEM import propagate_BEM_pressure, compute_E, BEM_gorkov_analytical
from acoustools.Solvers import gspat

import torch

pth = 'acoustools/tests/data/gcode/focal_point.lcode'
save_pth = 'acoustools/tests/data/gcode/focal_point.pth'
BEM_path='../BEMMedia'

board = TOP_BOARD

read_lcode(pth=pth, ids=(-1,), save_holo_name=save_pth)

x = pickle.load(open(save_pth,'rb'))[0]


reflector = load_scatterer(BEM_path+'/flat-lam2.stl')

p = create_points(1,1,0,0,0.01)
# E = compute_E(reflector, p, board, path=BEM_path)
# x = gspat(p,A=E, iterations=20)



abc = ABC(0.07)

Visualise(*abc, x,points=p, colour_functions=[propagate_BEM_pressure,], colour_function_args=[{'board':board, 'scatterer':reflector, "path":BEM_path}])
