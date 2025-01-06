from acoustools.Levitator import LevitatorController
from acoustools.BEM import compute_E, propagate_BEM_pressure
from acoustools.Mesh import load_scatterer
from acoustools.Utilities import create_points, TOP_BOARD
from acoustools.Solvers import wgs


root = "../BEMMedia/" #Change to path to BEMMedia Folder
path = root+"flat-lam2.stl"

reflector = load_scatterer(path,dz=-0.05) #Change dz to be the position of the reflector

board = TOP_BOARD
p = create_points(1,1,0,0,0) #point at (0,0,0)

E = compute_E(reflector, p, board, path=root)

x = wgs(p,A=E)

pressure = propagate_BEM_pressure(x,p,reflector,E=E)
print(pressure)

lev = LevitatorController(ids=-1) #Change to your board IDs
lev.levitate(x)
input("Press Enter to stop")
lev.disconnect()