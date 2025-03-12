import torch
from torch import Tensor
from acoustools.Utilities.Setup import device


def create_board(N:int, z:float) -> Tensor: 
    '''
    Create a single transducer array \n
    :param N: Number of transducers + 1 per side eg for 16 transducers `N=17`
    :param z: z-coordinate of board
    :return: tensor of transducer positions
    '''
    pitch=0.0105
    grid_vec=pitch*(torch.arange(-N/2+1, N/2, 1)).to(device)
    x, y = torch.meshgrid(grid_vec,grid_vec,indexing="ij")
    x = x.to(device)
    y= y.to(device)
    trans_x=torch.reshape(x,(torch.numel(x),1))
    trans_y=torch.reshape(y,(torch.numel(y),1))
    trans_z=z*torch.ones((torch.numel(x),1)).to(device)
    trans_pos=torch.cat((trans_x, trans_y, trans_z), axis=1)
    return trans_pos

# BOARD_POSITIONS = .234/2
BOARD_POSITIONS:float = 0.2365/2
'''
Static variable for the z-position of the boards, positive for top board, negative for bottom board
'''
  
def transducers(N=16,z=BOARD_POSITIONS) -> Tensor:
  '''
  :return: the 'standard' transducer arrays with 2 16x16 boards at `z = +-234/2 `
  '''
  return torch.cat((create_board(N+1,z),create_board(N+1,-1*z)),axis=0).to(device)



TRANSDUCERS:Tensor = transducers()
'''
Static variable for `transducers()` result
'''
TOP_BOARD:Tensor = create_board(17,BOARD_POSITIONS)
'''
Static variable for a 16x16 array at `z=.234/2` - top board of a 2 array setup
'''
BOTTOM_BOARD:Tensor = create_board(17,-1*BOARD_POSITIONS)
'''
Static variable for a 16x16 array at `z=-.234/2` - bottom board of a 2 array setup
'''
