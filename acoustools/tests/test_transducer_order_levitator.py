from acoustools.Utilities import TRANSDUCERS, forward_model, create_points, DTYPE, device, get_convert_indexes
from acoustools.Levitator import LevitatorController    

import torch

board = TRANSDUCERS
M = board.shape[0]

p = create_points(1,1,0,0,0)

IDS = get_convert_indexes(512)

F = forward_model(p, board)

lev = LevitatorController(ids=(73,53))
# lev = LevitatorControlsler(ids=-1)

disconnected = False

try:
    for i,transducer in enumerate(board):
        x = torch.zeros((1,M,1)).to(DTYPE).to(device)
        x[:,i,:] = 1
        print(i,transducer, torch.abs(F@x))
        lev.levitate(x)
        input()
        

except KeyboardInterrupt:
    lev.disconnect()
    disconnected= True

finally:
    if not disconnected:
        lev.disconnect()

