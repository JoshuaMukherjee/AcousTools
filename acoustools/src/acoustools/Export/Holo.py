'''
Export to .holo file -> List of holograms
'''

import pickle, torch
from acoustools.Utilities.Utilities import batch_list
from acoustools.Utilities.Setup import device,DTYPE
# from acoustools.Constants import wavelength

def compress(phase, levels=32):
    phase_divs = 2*3.1415/levels
    phase_levels = torch.angle(phase)/phase_divs
    phase_levels += levels / 2

    amp_divs = 1/levels
    amp_levels = torch.abs(phase)/amp_divs

    torch.round_(phase_levels).to(torch.int8)
    torch.round_(amp_levels).to(torch.int8)
    
    return phase_levels, amp_levels

def decompress(phases, amplitudes, levels=32):
    phase_divs = 2*3.1415/levels
    phases = [(p - levels/2) * phase_divs for p in phases]

    amp_divs = 1/levels
    amplitudes = [a * amp_divs for a in amplitudes]
    

    holo = [a*torch.e ** (1j*p) for a,p in zip(amplitudes,phases)]
    holo = torch.tensor(holo, device=device,dtype=DTYPE).unsqueeze_(0).unsqueeze_(2)
    return holo


def save_holograms(holos, fname):
    if '.' not in fname:
        fname += '.holo'
    # pickle.dump(holos, open(fname, 'wb'))
    
    with open(fname, 'wb') as file:
        for holo in holos:
            phase, amp = compress(holo.squeeze())
            for p in phase:
                p = int(p.item())
                file.write((p).to_bytes(6, byteorder='big', signed=False))
            file.write((2**6).to_bytes(6, byteorder='big', signed=False))
            for a in amp:
                a = int(a.item())
                file.write((a).to_bytes(6, byteorder='big', signed=False))
            file.write((2**6).to_bytes(6, byteorder='big', signed=False))

def load_holograms(path):
    if '.' not in path:
        path += '.holo'
    # holos = pickle.load(open(path, 'rb'))

    with open(path, 'rb') as file:
        phases = []
        amps = []
        holos = []
        reading_amps = 0
        data = file.read()
        for bits in batch_list(data,6):
            j = int.from_bytes(bits, byteorder='big')
            if j < 2**6:
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