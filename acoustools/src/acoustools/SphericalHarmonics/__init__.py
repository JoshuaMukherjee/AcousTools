from acoustools.Utilities import DTYPE
import torch


if DTYPE != torch.complex128:
    raise(RuntimeError("Spherical harmomics must be run using complex128 types due to a lack of accuracy at 64-bit data types. Please rerun using -c128 in the command line"))


from acoustools.SphericalHarmonics.SphericalHarmonics import *
from acoustools.SphericalHarmonics.Bessel import *
from acoustools.SphericalHarmonics.Propagators import n*