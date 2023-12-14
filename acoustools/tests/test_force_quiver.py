from acoustools.Visualiser import force_quiver
from acoustools.Mesh import load_scatterer,get_normals_as_points,get_centres_as_points, get_plane

if __name__ == "__main__":

    path = "Sphere-lam2.stl"
    scatterer = load_scatterer(path,dy=-0.06,root_path="../BEMMedia/") #Make mesh at 0,0,0

    origin = (0,0,0)
    normal = (0,1,0) 

    planar = get_plane(scatterer,origin,normal)


    norms = get_normals_as_points(planar)
    centres = get_centres_as_points(planar)

    pad = 0.005
    bounds = planar.bounds()
    xlim=[bounds[0]-pad,bounds[1]+pad]
    ylim=[bounds[2]-pad,bounds[3]+pad]

    force_quiver(centres,norms[:,0,:],norms[:,2,:], normal,xlim,ylim)
