'''
Export to pickled file -> List of holograms
Mostly for backwards compatibility - not imported into __init__
'''

import pickle

def save_holograms(holos, path):
    pickle.dump(holos, open(path, 'wb'))


def load_holograms(path):
    holos = pickle.load(open(path, 'rb'))
    return holos