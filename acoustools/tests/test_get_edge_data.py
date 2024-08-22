from acoustools.Mesh import load_scatterer, get_edge_data


path = '../BEMMedia/'
scatterer = load_scatterer('Hand-0-lam2.STL', root_path=path)
get_edge_data(scatterer)