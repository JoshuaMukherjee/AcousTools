from acoustools.Fabrication.Slicer import slice_mesh
from acoustools.Mesh import load_scatterer, get_centres_as_points
from acoustools.Paths import interpolate_path_to_distance

import matplotlib.pyplot as plt

import vedo
import vedo.pointcloud

path = "../BEMMedia"
bunny = load_scatterer(path+"/Bunny-lam2.stl")

# vedo.show(vedo.Points(bunny))


layers = slice_mesh(bunny)
layer1 = layers[1]
points = get_centres_as_points(layer1)
xs = points[:,0,:].cpu().detach().tolist()
ys = points[:,1,:].cpu().detach().tolist()

plt.scatter(xs,ys) #Would neeed to have travelling salesman to get right order - then use that to interpolate

plt.show()

# vedo.show(*layers)
