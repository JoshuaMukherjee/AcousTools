'''
Export to .holo file -> List of holograms
'''

import pickle

def save_holograms(holos, fname):
    if '.' not in fname:
        fname += '.holo'
    pickle.dump(holos, open(fname, 'wb'))


def load_holograms(path):
    if '.' not in path:
        path += '.holo'
    holos = pickle.load(open(path, 'rb'))
    return holos