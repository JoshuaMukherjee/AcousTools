from acoustools.Intepreter import read_lcode
from acoustools.BEM import propagate_BEM_pressure
from acoustools.Visualiser import Visualise, ABC
from acoustools.Mesh import load_scatterer
from acoustools.Utilities import TOP_BOARD, create_points


root = "../BEMMedia/" #Change to path to BEMMedia Folder
path = root+"flat-lam2.stl"

reflector = load_scatterer(path) #Change dz to be the position of the reflector


board = TOP_BOARD

p = create_points(1,1,0.0,0.0,0.03)


pth = 'acoustools/tests/data/gcode/gorkov_target.lcode'
xs = read_lcode(pth=pth, ids=(-1,), return_holos=True)

x = xs[0]

abc = ABC(0.05)
Visualise(*abc, x,p, colour_functions=[propagate_BEM_pressure], colour_function_args=[{'scatterer':reflector,'path':root}])