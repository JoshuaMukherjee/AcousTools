from acoustools.Utilities import TRANSDUCERS, forward_model, create_points, DTYPE, device, get_convert_indexes

import torch

board = TRANSDUCERS
M = board.shape[0]

p = create_points(1,1,0,0,0)

IDS = get_convert_indexes(512)

# permuted_board = board[IDS]


F = forward_model(p, board)


for i,transducer in enumerate(board):
    x = torch.zeros((M,1)).to(DTYPE).to(device)
    x[i,:] = 1

    print(i,transducer, torch.abs(F@x))
