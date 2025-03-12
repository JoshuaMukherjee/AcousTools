import torch, math, sys
import acoustools.Constants as Constants

torch.cuda.empty_cache()

from typing import Literal
from types import FunctionType
from torch import Tensor

DTYPE = torch.complex64
'''
Data type to use for matricies - use `.to(DTYPE)` to convert
'''

device:Literal['cuda','cpu'] = 'cuda' if torch.cuda.is_available() else 'cpu' 
'''Constant storing device to use, `cuda` if cuda is available else cpu. \n
Use -cpu when running python to force cpu use'''
device = device if '-cpu' not in sys.argv else 'cpu'


from acoustools.Utilities.Boards import *
from acoustools.Utilities.Export import *
from acoustools.Utilities.Forward_models import *
from acoustools.Utilities.Piston_model_gradients import *
from acoustools.Utilities.Points import *
from acoustools.Utilities.Propagators import *
from acoustools.Utilities.Signatures import *
from acoustools.Utilities.Targets import *
from acoustools.Utilities.Utilities import *
