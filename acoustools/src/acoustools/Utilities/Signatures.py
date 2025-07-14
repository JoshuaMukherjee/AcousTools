
from torch import Tensor
import torch

from typing import Literal

from acoustools.Utilities.Boards import TRANSDUCERS
from acoustools.Utilities.Setup import device

def add_lev_sig(activation:Tensor, board:Tensor|None=None, 
                mode:Literal['Focal', 'Trap', 'Vortex','Twin', 'Eye']='Trap', sig:Tensor|None=None, return_sig:bool=False, board_size:int=256) -> Tensor:
    '''
    Adds signature to hologram for a board \n
    :param activation: Hologram input
    :param board: Board to use
    :param mode: Type of signature to add, should be one of
    * Focal: No signature
    * Trap: Add $\\pi$ to the top board - creates a trap
    * Vortex: Add a circular signature to create a circular trap
    * Twin: Add $\\pi$ to half of the board laterally to create a twin trap
    * Eye: Add a vortex trap combined with a central disk of the Trap method. Produces a eye like shape around the focus
    :param sig: signature to add to top board. If `None` then value is determined by value of `mode`
    :param board_size: Transducers per board in total. Default 256 (16x16)
    :return: hologram with signature added

    ```Python
    from acoustools.Utilities import create_points, add_lev_sig
    from acoustools.Solvers import wgs

    p = create_points(1,x=0,y=0,z=0)
    x = wgs(p, board=board)
    x_sig, sig = add_lev_sig(x.clone(), mode=mode, return_sig=True, board=board)

    ```
    '''
    if board is None:
        board = TRANSDUCERS

    act = activation.clone().to(device)

    s = act.shape
    B = s[0]

    act = torch.reshape(act,(B,-1, board_size))

    # act[:,0,:] = torch.e**(1j*(sig + torch.angle(act[:,0,:].clone())))
    if sig is None:
        sig = torch.ones_like(act)
        if mode == 'Trap':
            sig = torch.stack([torch.ones_like(act[:,0,:]) * torch.pi, torch.zeros_like(act[:,0,:])],dim=1)
        if mode == 'Focal':
            sig = torch.zeros_like(act)
        if mode == 'Vortex':
            plane = board[:,0:2]
            sig = torch.atan2(plane[:,0], plane[:,1]).unsqueeze(0).unsqueeze(2).reshape((B,-1, board_size))
        if mode == 'Twin':
            plane = board[:,0:2]
            sig = torch.zeros_like(sig) + torch.pi * (plane[:,0] > 0).unsqueeze(0).unsqueeze(2).reshape((B,-1, board_size))
        if mode == 'Eye':
            
            b = board.reshape(-1,board_size,3)

            plane = board[:,0:2]
            sig = torch.atan2(plane[:,0], plane[:,1]).unsqueeze(0).unsqueeze(2).reshape((B,-1, board_size))
            mask = torch.sqrt(b[:,:,0] ** 2 + b[:,:,1] ** 2) < 0.06

            

            sig[0,0,:][mask[0,:] == 1] = torch.pi
            sig[0,1,:][mask[0,:] == 1] = 0

    x = torch.abs(act) * torch.exp(1j* (torch.angle(act) + sig))

    x = torch.reshape(x,s)

    if return_sig:
        return x, sig
    return x





