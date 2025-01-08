from acoustools.Utilities import TRANSDUCERS, get_convert_indexes, TOP_BOARD, BOTTOM_BOARD

if __name__ == "__main__":
    IDX = get_convert_indexes(256,'top')
    # print(IDX)
    flip = TOP_BOARD[IDX]
    for i,row in enumerate(flip):
        print(i+1, row)