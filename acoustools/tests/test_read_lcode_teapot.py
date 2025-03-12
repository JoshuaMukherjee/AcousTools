from acoustools.Intepreter import read_lcode
from acoustools.Mesh import load_scatterer, scale_to_diameter
import vedo

pth = 'acoustools/tests/data/gcode/teapot.lcode'
msh = load_scatterer("../BEMMedia/Teapot.stl")

scale_to_diameter(msh, 0.1) #NOT TO SCALE TO gcode FILE, FOR EXAMPLE USE ONLY

read_lcode(pth=pth, ids=(-1,), mesh=msh)
