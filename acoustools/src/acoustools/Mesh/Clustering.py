
from vedo import Mesh
from torch_kmeans import KMeans

from acoustools.Mesh.Features import get_centres_as_points, get_normals_as_points

import torch

def k_means_cluster(scatterer: Mesh, k = 8, starting_means = None, clusterer:KMeans = None, init_method='rnd', max_iter = 100, 
                    scatterer_centres = None, scatterer_normals = None, use_centres=True, use_normals = True, return_centroids=False):
    
    if clusterer is None:
        clusterer = KMeans(init_method=init_method, max_iter=max_iter, num_init=k, n_clusters=k, verbose=False)
        
    if scatterer_centres is None:
        scatterer_centres = get_centres_as_points(scatterer) 
    # print(scatterer_centres)
    # exit()
    
    if use_normals and scatterer_normals is None:
        scatterer_normals = get_normals_as_points(scatterer)
    
    data = []
    if use_normals:
        data.append(scatterer_normals)
    if use_centres:
        data.append(scatterer_centres)
        
    data = torch.cat(data, dim=1)
    
    data = data.permute(0,2,1).real
    result = clusterer(data)
    clusters = result[0]
    centroids = result[1]
    if return_centroids:
        return clusters, centroids
    
    return clusters
    
    
    
    
    