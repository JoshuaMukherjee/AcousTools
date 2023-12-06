from acoustools.Utilities import create_points,TRANSDUCERS
from acoustools.Mesh import filter_transducers_forward_model

if __name__ == "__main__":
    p = create_points(10,2)

    filter_transducers_forward_model(p,TRANSDUCERS)
    