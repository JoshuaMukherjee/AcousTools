from acoustools.Utilities import create_points, TRANSDUCERS, add_lev_sig
from acoustools.Solvers import wgs
from acoustools.Visualiser import get_image_positions, ABC, force_quiver
from acoustools.Force import compute_force

board = TRANSDUCERS

p = create_points(1,1,0,0,0)
x = wgs(p,board=board)
x = add_lev_sig(x)

abc = ABC(0.01, plane = 'xz')
res = (20,20)
positions = get_image_positions(*abc, res=res)
force_x, force_y, force_z = compute_force(x, positions, board=board, return_components=True)

force_quiver(positions, -1*force_x, force_z)