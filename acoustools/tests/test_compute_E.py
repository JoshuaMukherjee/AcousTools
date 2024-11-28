from acoustools.Mesh import load_scatterer
from acoustools.Utilities import create_points, TOP_BOARD
from acoustools.BEM import compute_E


path = r"C:\Users\joshu\Documents\BEMMedia\Bunny-lam2.stl"

reflector = load_scatterer(path,dz=-0.05)

p = create_points(1,1,x=0,y=0,z=0)


E = compute_E(reflector,points=p, board=TOP_BOARD,path=r"C:\Users\joshu\Documents\BEMMedia", use_cache_H=False)