    
# This is ....
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import arcpy
import os
import scipy as scipy
import pandas as pd
from scipy.linalg import eig
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import os

def calculate_landscapematrix(file_path = FilePath, use_centroid=True, decay_factor=0.5):
    # Function to calculate distance matrix
    def distanceMatrix(xy):
        return squareform(pdist(xy))

    # Function to calculate polygon distances
    def polygon_distance_matrix(gdf):
        n = len(gdf)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = gdf.geometry.iloc[i].distance(gdf.geometry.iloc[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        return dist_matrix

    # Function to read the feature class based on file extension
    def read_feature_class(file_path):
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.gpkg':
            gdf = gpd.read_file(file_path, layer=0)
        elif file_extension == '.gdb':
            #gdf = gpd.read_file(file_path, driver='FileGDB', layer="GRUK")
            gdf = gpd.read_file(file_path, driver='FileGDB', layer="0")
        else:
            raise ValueError("Unsupported file format. Please provide a GeoPackage (.gpkg) or File Geodatabase (.gdb).")
        return gdf

    # Read the feature class
    gdf = read_feature_class(file_path)

    if use_centroid:
        # Calculate the centroid coordinates
        gdf['coordinates'] = gdf.geometry.centroid
        xy = np.array(list(zip(gdf.coordinates.x, gdf.coordinates.y)))
        # Ensure xy is a 2-dimensional array
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Coordinates array 'xy' must be a 2-dimensional array with shape (n, 2).")
        # Calculate distance matrix using centroids
        dists = distanceMatrix(xy)
    else:
        # Calculate distance matrix using polygons
        dists = polygon_distance_matrix(gdf)

    # Calculate the areas of the geometries
    sizes = gdf.geometry.area

    # Create Outer Matrix
    areas_array = sizes
    outer_matrix = np.outer(areas_array, areas_array)  # Calculate the outer product matrix

    # Calculate realized distances with the decay factor
    realizedDists = np.exp(-dists * decay_factor)

    # Calculate Mij
    Mij = realizedDists * outer_matrix

    # Set diagonal to zero
    np.fill_diagonal(Mij, 0)

    # return results
    return Mij

    # Print the result
    print(Mij)    

def calculate_eigenmeasure(LandscapeMatrix=None, geometry=None, use_centroid=True, decay_factor=0.5, IDs=None):
    if LandscapeMatrix is None and geometry is not None:
        LandscapeMatrix = calculate_matrix(geometry, use_centroid, decay_factor)
    
    if LandscapeMatrix is None:
        raise ValueError("Either LandscapeMatrix or geometry must be provided.")
    
    # If IDs is not provided, create an enumerated index
    if IDs is None:
        IDs = np.arange(LandscapeMatrix.shape[0])
    
    # Define the function to calculate contribution
    def calculate_contribution(Lambda, Mij):
        num_patches = Mij.shape[0]
        contributions = np.zeros(num_patches)
        
        for ii in range(num_patches):
            # Exclude the ii-th row and column from Mij
            Mij_sub = np.delete(Mij, ii, axis=0)
            Mij_sub = np.delete(Mij_sub, ii, axis=1)
            
            # Calculate the largest eigenvalue of the modified Mij
            eigenvalues = np.linalg.eigvals(Mij_sub)
            largest_eigenvalue = np.max(eigenvalues)
            
            # Calculate the contribution
            contributions[ii] = (Lambda - largest_eigenvalue) / Lambda
        
        return contributions

    eigenvalues, eigenvectors = scipy.linalg.eig(LandscapeMatrix)
    Le = np.argmax(eigenvalues)
    Lambda = np.real(eigenvalues[Le])
    
    # Approximate stable age distribution = right eigenvector
    w0 = np.real(eigenvectors[:, Le])
    w = np.abs(w0)
    # Reproductive value = left eigenvector
    # This equals the right eigenvector of the landscape matrix transposed
    V = np.linalg.inv(eigenvectors).conj()
    v = np.abs(np.real(V[Le, ]))
    
    # Contribution of the patch to lambda
    # When considering small perturbations, loses a small part of the habitats and habitat degradation
    pc = v.T * w  # These are normalized

    # Call the function with your Lambda and Mij data
    contributions = calculate_contribution(Lambda, LandscapeMatrix)

    # Collect all the measures
    EigenMeasures = pd.DataFrame({"ID": IDs,
                                  "PC_small": pc,
                                  "PC_large": contributions,
                                  "REv": w, 
                                  "LEv": v})
    
    return EigenMeasures

def communities(LandscapeMatrix=None, geometry=None, use_centroid=True, decay_factor=0.5):
    if LandscapeMatrix is None and geometry is not None:
        LandscapeMatrix = calculate_matrix(geometry, use_centroid, decay_factor)
    
    if LandscapeMatrix is None:
        raise ValueError("Either LandscapeMatrix or geometry must be provided.")
    
    # Convert the input matrix to a NumPy array
    M = np.array(LandscapeMatrix)
    # Calculate the total number of edges/links in the landscape
    m = np.sum(M > 0)
    
    # Calculate the degrees of the vertices/nodes/patches
    kj = np.sum(M > 0, axis=0)
    ki = np.sum(M > 0, axis=1)
    
    # Define the modularity matrix
    B = M - np.outer(ki, kj) / (2 * m)
    # Compute the eigenvalues and eigenvectors of the modularity matrix
    ev = np.linalg.eig(B)
    
    # Find the index of the largest eigenvalue
    lmax = np.argmax(np.real(ev[0]))
    
    # Get the eigenvector corresponding to the largest eigenvalue
    W = ev[1][:, lmax]
    
    # Create a frame with indices and group labels based on the sign of the eigenvector
    w_frame = np.column_stack((np.arange(len(W)), np.array([chr(97 + int(x)) for x in np.sign(W) + 1])))
    # Generate positions for the nodes for visualization
    #pos = nx.spring_layout(nx.from_numpy_matrix(M))
    
    # Plot the nodes with colors based on their group labels
    #plt.scatter(*zip(*pos.values()), c=[ord(x) for x in w_frame[:, 1]])
    #plt.show()
    print("Run loop")
    t = 0
    while len(np.unique(w_frame[:, t])) != len(np.unique(w_frame[:, t + 1])):
        new_col = np.empty(len(w_frame), dtype=object)
        
        for i in np.unique(w_frame[:, t + 1]):
            idx = np.where(w_frame[:, t + 1] == i)[0]
            subB = B[np.ix_(idx, idx)]
            ev = np.linalg.eig(subB)
            lmax = np.argmax(np.real(ev[0]))
            W = ev[1][:, lmax]
            new_col[idx] = [f"{w_frame[idx[0], t + 1]}-{chr(97 + int(x))}" for x in np.sign(W) + 1]
        
        w_frame = np.column_stack((w_frame, new_col))
        #w_frame[:, t + 1] = np.array([ord(x) for x in w_frame[:, t + 1]])
        # Convert the first column of characters to numbers starting from 1
        unique_chars, indices = np.unique(w_frame[:, t + 1], return_inverse=True)
        w_frame[:, t + 1] = indices + 1
        t += 1
    
    return w_frame[:, :-1]
