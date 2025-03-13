import torch, sys
from typing import Literal 

DTYPE = torch.complex32
'''
Data type to use for matricies - use `.to(DTYPE)` to convert
'''

device:Literal['cuda','cpu'] = 'cuda' if torch.cuda.is_available() else 'cpu' 
'''Constant storing device to use, `cuda` if cuda is available else cpu. \n
Use -cpu when running python to force cpu use'''
device = device if '-cpu' not in sys.argv else 'cpu'
