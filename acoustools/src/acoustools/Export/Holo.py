'''
Export to .holo file -> List of holograms\n
The holo format stores each phase and amplitude as a 5 bit integer with each phase and ampplitude being discritised to 32 levels \n
The result of this is then compressed using pythons zlib (https://docs.python.org/3/library/zlib.html)
Internally stored as blocks of phase and amplitude seperated by 2^6-1 eg phase1 <11111> amplitude1 <11111> phase2 <11111> amplitude2 <11111> ...
'''

import torch, io, zlib
from acoustools.Utilities.Utilities import batch_list
from acoustools.Utilities.Setup import device,DTYPE
# from acoustools.Constants import wavelength

from torch import Tensor

def compress(hologram:Tensor, levels:int=32) -> list[Tensor]:
    '''
    @private
    '''
    phase_divs = 2*3.1415/levels
    phase_levels = torch.angle(hologram)/phase_divs
    phase_levels += levels / 2

    amp_divs = 1/levels
    amp_levels = torch.abs(hologram)/amp_divs

    torch.round_(phase_levels).to(torch.int8)
    torch.round_(amp_levels).to(torch.int8)
    
    return phase_levels, amp_levels

def decompress(hologram:list, amplitudes:list, levels:int=32) -> list[Tensor]:
    '''
    @private
    '''
    phase_divs = 2*3.1415/levels
    phases = [(p - levels/2) * phase_divs for p in hologram]

    amp_divs = 1/levels
    amplitudes = [a * amp_divs for a in amplitudes]
    

    holo = [a*torch.e ** (1j*p) for a,p in zip(amplitudes,phases)]
    holo = torch.tensor(holo, device=device,dtype=DTYPE).unsqueeze_(0).unsqueeze_(2)
    return holo


def save_holograms(holos:list[Tensor]|Tensor, fname:str):
    '''
    Save holograms in .holo format. 
    :param holos: Holograms to use
    :param fname:  filename to use. Will append .holo is no extension provides
    '''
    if '.' not in fname:
        fname += '.holo'
    # pickle.dump(holos, open(fname, 'wb'))
    
    # with open(fname, 'wb') as file:
    with io.BytesIO() as file:
        for holo in holos:
            phase, amp = compress(holo.squeeze())
            for p in phase:
                p = int(p.item())
                file.write((p).to_bytes(5, byteorder='big', signed=False))
            file.write((2**6-1).to_bytes(5, byteorder='big', signed=False))
            for a in amp:
                a = int(a.item())
                file.write((a).to_bytes(5, byteorder='big', signed=False))
            file.write((2**6-1).to_bytes(5, byteorder='big', signed=False))
        
        new_data = zlib.compress(file.getbuffer())
        f = open(fname, 'wb')
        f.write(new_data)
        f.close()

def load_holograms(path:str) -> list[Tensor]:
    '''
    Reads .holo file
    :param path: Path to read
    :returns holgrams:
    '''
    if '.' not in path:
        path += '.holo'
    # holos = pickle.load(open(path, 'rb'))

    with open(path, 'rb') as file:
        phases = []
        amps = []
        holos = []
        reading_amps = 0
        data = file.read()
        data = zlib.decompress(data)
        for bits in batch_list(data,5):
            j = int.from_bytes(bits, byteorder='big')
            if j < 2**6-1:
                if not reading_amps:
                    phases.append(j)
                else:
                    amps.append(j)
            else:
                if reading_amps == 0:
                    reading_amps = 1
                else:
                    reading_amps = 0
                    holos.append([phases, amps])
                    phases = []
                    amps = []
        xs = []
        for h in holos:
            x = decompress(h[0], h[1])
            xs.append(x)

    return xs