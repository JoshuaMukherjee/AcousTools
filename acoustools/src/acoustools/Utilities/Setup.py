import torch, sys
from typing import Literal 

DTYPE = torch.complex64
DTYPE = torch.complex32 if '-c32' in sys.argv else DTYPE
DTYPE = torch.complex64 if '-c64' in sys.argv else DTYPE
DTYPE = torch.complex128 if '-c128' in sys.argv else DTYPE
'''
Data type to use for matricies - use `.to(DTYPE)` to convert
'''

device:Literal['cuda','cpu'] = 'cuda' if torch.cuda.is_available() else 'cpu' 
# device = 'mps' if torch.backends.mps.is_available() else device
'''Constant storing device to use, `cuda` if cuda is available else cpu. \n
Use -cpu when running python to force cpu use'''
device = device if '-cpu' not in sys.argv else 'cpu'
device = device if '-mps' not in sys.argv else 'mps'
