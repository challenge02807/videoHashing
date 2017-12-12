import numpy as np
import pickle
from adjusted_rand_index import rand_index
from sklearn.cluster import AgglomerativeClustering
import time

# Record the time of the clustering
t0 = time.time()

# Import data from the stored pickle file
full_data_feat = []
full_data_name = []
print("Getting hashed data...")
with open('data_6_phash_36.p', 'rb') as fp:
    data = pickle.load(fp)
    for name,features in data.items():
        full_data_name.append(name) # Store the name of the video
        bin_str = "" # The accumulating bit string
        for feat in features:
            bin_str += ('{:036b}'.format(feat)) # Append the next feature in binary representation
        full_data_feat.append([int(i) for i in bin_str]) # Store the bit string as an array of one and zeros

# Convert to numpy
full_data_feat = np.array(full_data_feat)
full_data_name = np.array(full_data_name)

print("Using "+str(full_data_feat.shape[1])+" features.")

# Compute the similarity matrix by using hamming distance
# Use matrix broadcasting
print("Computing distances...")
sim_matrix = (full_data_feat[:, None, :] != full_data_feat).sum(2)

# Bottom up hierarchical clustering with complete/maximal cluster distance
print("Clustering data...")
model = AgglomerativeClustering(n_clusters=970,linkage="complete",affinity="precomputed")
# Input the similarity matrix and fit the data to the model
model.fit(sim_matrix)

# Print snippets of the assigned clusters
print(model.labels_)

# For each video get the cluster id and
# store the name of the video in a set
clusters = [-1]*970
for i, name in enumerate(full_data_name):
    cluster_idx = model.labels_[i]
    if clusters[cluster_idx] == -1: # No name has been assigned yet.
        clusters[cluster_idx] = {name.split('.')[0]}
    else: # Add to previous set
        clusters[cluster_idx].add(name.split('.')[0])

# Compute adjusted rand index and report the time used
print("Final rand index = ",rand_index(clusters))
print("Query took %0.2f seconds" % (time.time()-t0))