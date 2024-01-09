
from acoustools.Mesh import get_centres_as_points, load_multiple_scatterers, load_scatterer, scale_to_diameter, merge_scatterers
from acoustools.Utilities import TRANSDUCERS, get_rows_in

if __name__ == "__main__":

    board = TRANSDUCERS

    wall_paths = ["../BEMMedia/flat-lam1.stl","../BEMMedia/flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.06,0.06],rotys=[90,-90]) #Make mesh at 0,0,0
    
    
    ball_path = "../BEMMedia/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.04)

    scatterer = merge_scatterers(ball, walls)

    scatterer_cells = get_centres_as_points(scatterer)
    ball_cells = get_centres_as_points(ball)

    mask = get_rows_in(scatterer_cells, ball_cells)
    print(mask.shape)
    print(scatterer_cells.shape)

    B = scatterer_cells.shape[0]
    print(scatterer_cells)
    print(scatterer_cells[mask].reshape(B,3,-1))



