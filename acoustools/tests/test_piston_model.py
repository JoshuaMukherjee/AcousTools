from acoustools.Utilities import forward_model, create_points, TRANSDUCERS

N_z = 4 #Number of points
board = TRANSDUCERS #2x16x16 Transducers
p = create_points(N_z)
F = forward_model(p, board)
