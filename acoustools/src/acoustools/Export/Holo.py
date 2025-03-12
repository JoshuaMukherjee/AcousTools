'''
Export to .holo file -> List of holograms
'''

import pickle

def save_holograms(holos, path):
    pickle.dump(holos, open(path, 'wb'))


def load_holograms(path):
    holos = pickle.load(open(path, 'rb'))
    return holos