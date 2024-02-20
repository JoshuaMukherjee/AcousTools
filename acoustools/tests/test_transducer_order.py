from acoustools.Utilities import TRANSDUCERS, get_convert_indexes

if __name__ == "__main__":
    IDX = get_convert_indexes()
    print(IDX)
    flip = TRANSDUCERS[IDX]
    for row in flip:
        print(row)