from acoustools.Mesh import load_scatterer, get_normals_as_points, get_centres_as_points, get_areas
from acoustools.Utilities import TRANSDUCERS
from acoustools.Force import force_mesh
from acoustools.Solvers import wgs
from acoustools.BEM import get_cache_or_compute_H, BEM_forward_model_grad, compute_E

import torch
import matplotlib.pyplot as plt

path = "../BEMMedia"

USE_CACHE = True
board = TRANSDUCERS

sphere_pth =  path+"/Sphere-lam2.stl"
sphere = load_scatterer(sphere_pth, dy=-0.06, dz=0.0) #Make mesh at 0,0,0

centres = get_centres_as_points(sphere)
areas = get_areas(sphere)

H = get_cache_or_compute_H(sphere, board, path=path)
x = wgs(centres, A=H)

normals = get_normals_as_points(sphere)

factor = 1000
forcesX = []
forcesY = []
forcesZ = []
scales = []
for i in range(10):

    points = centres + i/factor * normals
    norms = normals + i/factor * normals

    force = force_mesh(x, points, norms, areas,board, BEM_forward_model_grad, 
                    grad_function_args = {'scatterer':sphere,'H':H}, 
                    F_fun=compute_E, F_function_args={'scatterer':sphere,'H':H, 'board':board})
    
    scales.append(i/factor)
    net = torch.sum(force, axis=2).detach().squeeze()
    forcesX.append(net[0])
    forcesY.append(net[1])
    forcesZ.append(net[2])

plt.plot(scales, forcesX)
plt.plot(scales, forcesY)
plt.plot(scales, forcesZ)

plt.show()
