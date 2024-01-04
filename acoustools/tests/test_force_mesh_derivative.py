from acoustools.Gorkov import force_mesh_derivative
from acoustools.BEM import  get_cache_or_compute_H
from acoustools.Mesh import load_scatterer,scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas
from acoustools.Utilities import TRANSDUCERS
from acoustools.Solvers import wgs_wrapper



if __name__ == "__main__":
    ball_path = "../BEMMedia/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.04)
    
    centres = get_centres_as_points(ball)

    H = get_cache_or_compute_H(ball,TRANSDUCERS,path = "../BEMMedia/")
    x = wgs_wrapper(centres,A=H)

    norms = get_normals_as_points(ball)
    areas = get_areas(ball)

    
    Fa = force_mesh_derivative(x,centres,norms,areas,TRANSDUCERS,ball)

    print(Fa)