from acoustools.BEM import grad_2_H, get_cache_or_compute_H
from acoustools.Mesh import load_scatterer,scale_to_diameter, get_centres_as_points
from acoustools.Utilities import TRANSDUCERS
from acoustools.Solvers import wgs_wrapper


if __name__ == "__main__":
    ball_path = "../BEMMedia/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.04)
    
    centres = get_centres_as_points(ball)
    H = get_cache_or_compute_H(ball,TRANSDUCERS,path = "../BEMMedia/")
    Haa = grad_2_H(None,ball,TRANSDUCERS)
    x = wgs_wrapper(centres,A=H)

    print(Haa@x)

