from acoustools.Mesh import load_scatterer, translate, get_centres_as_points
from acoustools.BEM import get_cache_or_compute_H, get_cache_or_compute_H_gradients, grad_H
from acoustools.Utilities import TRANSDUCERS, propagate
from acoustools.Solvers import wgs
from acoustools.Constants import wavelength

import torch

torch.set_printoptions(linewidth=100)

path = "../BEMMedia"
paths = path+"/Sphere-lam2.stl"
ball = load_scatterer(paths, dy=-0.06)

cache = False

board = TRANSDUCERS

H = get_cache_or_compute_H(ball,board, path=path,use_cache_H=cache)
Hx, Hy, Hz = grad_H(None, ball, board,path=path, use_cache_H=cache, H=H)


centres = get_centres_as_points(ball)
x = wgs(centres,A=H)
pressure = propagate(x,centres,board,A=H)
# print(pressure)

h = wavelength/32

ball_dx_left = ball.clone(deep=True)
translate(ball_dx_left, dx=-1*h)
H_dx_left = get_cache_or_compute_H(ball_dx_left,board, path=path,use_cache_H=cache)
pressure_dx_left  = propagate(x,get_centres_as_points(ball_dx_left),board,A=H_dx_left)


ball_dx_right = ball.clone(deep=True)
translate(ball_dx_right, dx=h)
H_dx_right = get_cache_or_compute_H(ball_dx_right,board, path=path,use_cache_H=cache)
pressure_dx_right  = propagate(x,get_centres_as_points(ball_dx_right),board,A=H_dx_right)


grad_H_x = (pressure_dx_right - pressure_dx_left) / (2*h + 0j)

Px_fd = torch.abs(grad_H_x).squeeze()
Px_an = torch.abs(Hx@x).squeeze()

print(torch.stack([Px_fd, Px_an, Px_fd/Px_an]))
# print(grad_H_x, Hx@x)


ball_dy_left = ball.clone(deep=True)
translate(ball_dy_left, dy=-1*h)
H_dy_left = get_cache_or_compute_H(ball_dy_left,board, path=path,use_cache_H=cache)
pressure_dy_left  = propagate(x,get_centres_as_points(ball_dy_left),board,A=H_dy_left)


ball_dy_right = ball.clone(deep=True)
translate(ball_dy_right, dy=h)
H_dy_right = get_cache_or_compute_H(ball_dy_right,board, path=path,use_cache_H=cache)
pressure_dy_right  = propagate(x,get_centres_as_points(ball_dy_right),board,A=H_dy_right)

grad_H_y = (pressure_dy_right - pressure_dy_left) / (2*h + 0j)
Py_fd = torch.abs(grad_H_y).squeeze()
Py_an = torch.abs(Hy@x).squeeze()

print(torch.stack([Py_fd, Py_an, Py_fd/Py_an]))



ball_dz_left = ball.clone(deep=True)
translate(ball_dz_left, dz=-1*h)
H_dz_left = get_cache_or_compute_H(ball_dz_left,board, path=path,use_cache_H=cache)
pressure_dz_left  = propagate(x,get_centres_as_points(ball_dz_left),board,A=H_dz_left)


ball_dz_right = ball.clone(deep=True)
translate(ball_dz_right, dz=h)
H_dz_right = get_cache_or_compute_H(ball_dz_right,board, path=path,use_cache_H=cache)
pressure_dz_right  = propagate(x,get_centres_as_points(ball_dz_right),board,A=H_dz_right)

grad_H_z = (pressure_dz_right - pressure_dz_left) / (2*h + 0j)
Pz_fd = torch.abs(grad_H_z).squeeze()
Pz_an = torch.abs(Hz@x).squeeze()

print(torch.stack([Pz_fd, Pz_an, Pz_fd/Pz_an]))




