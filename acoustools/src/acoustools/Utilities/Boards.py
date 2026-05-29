import torch
from torch import Tensor
from acoustools.Utilities.Setup import device, deprecated, supress_warnings
from acoustools.Constants import pitch

import warnings


def create_board(N:int, z:float, pitch:float = pitch) -> Tensor: 
    '''
    Create a single transducer array \n
    :param N: Number of transducers per side eg for 16 transducers `N=16` NOTE: before AcousTools 1.1.2 this behavioour was different. In the past N-1 Transducers were created. Run using --deprecated for old bahaviour
    :param z: z-coordinate of board
    :param pitch: The pitch of the transducers (gap between them)
    :return: tensor of transducer positions
    '''

    if N == 17 and not supress_warnings: warnings.warn(
      'acoustools.Boards.create_board: behavious has changed from acoustools 1.1.2 onwards.' \
      ' N equals number of transducers, triggered by requesting baord of size 17 (the parameter to get a standard 16x16 PAT under old bahviour) ' \
      '- if a 17x17 array  is required then ignore this warning and run AcousTools with --supress_warninigs'
    )
    
    if not deprecated:
       N = N + 1
    else:
       if not supress_warnings: warnings.warn(
        'acoustools.Boards.create_board: running in deprecated mode -' \
        ' old behavious is for N to be number of transducers+1, ' \
        'new behaviour is N equals number of transducers'
       )
    
       
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
  return torch.cat((create_board(N,z),create_board(N,-1*z)),axis=0).to(device)



TRANSDUCERS:Tensor = transducers()
'''
Static variable for `transducers()` result
'''
TOP_BOARD:Tensor = create_board(16,BOARD_POSITIONS)
'''
Static variable for a 16x16 array at `z=.234/2` - top board of a 2 array setup
'''
BOTTOM_BOARD:Tensor = create_board(16,-1*BOARD_POSITIONS)
'''
Static variable for a 16x16 array at `z=-.234/2` - bottom board of a 2 array setup
'''
