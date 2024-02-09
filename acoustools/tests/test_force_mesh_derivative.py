from acoustools.Force import force_mesh_derivative
from acoustools.BEM import  get_cache_or_compute_H
from acoustools.Mesh import load_scatterer,scale_to_diameter, get_centres_as_points, get_normals_as_points, get_areas, load_multiple_scatterers, merge_scatterers
from acoustools.Utilities import TRANSDUCERS
from acoustools.Solvers import wgs_wrapper



if __name__ == "__main__":
    path = "../../BEMMedia"
    wall_paths = [path+"/flat-lam1.stl",path+"flat-lam1.stl"]
    walls = load_multiple_scatterers(wall_paths,dxs=[-0.06,0.06],rotys=[90,-90]) #Make mesh at 0,0,0
    
    ball_path = path+"/Sphere-lam2.stl"
    ball = load_scatterer(ball_path,dy=-0.06) #Make mesh at 0,0,0
    scale_to_diameter(ball,0.04)
    
    scatterer = merge_scatterers(ball, walls)

    centres = get_centres_as_points(scatterer)

    H = get_cache_or_compute_H(scatterer,TRANSDUCERS,path = path)
    x = wgs_wrapper(centres,A=H)

    norms = get_normals_as_points(scatterer)
    areas = get_areas(scatterer)

    
    Fa = force_mesh_derivative(x,centres,norms,areas,TRANSDUCERS,scatterer)

    print(Fa)