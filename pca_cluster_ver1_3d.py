import pandas as pd
import glob
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Read all csv files.
file_paths = glob.glob('xrd_dirac/*.csv')

# Read spectrum data from each csv file.
xrd_dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Find peaks in all XRD patterns.
peak_data = []
for df in xrd_dataframes:
    y_values = df['y'].values
    peaks, _ = find_peaks(y_values, height=0)  # height=0: find all peaks
    peak_data.append(y_values[peaks])

# Find the spectrum with the maximum number of peaks.
max_len = max(len(data) for data in peak_data)

# Pad the peak data of all spectra to have the same length.
for i, data in enumerate(peak_data):
    if len(data) < max_len:
        peak_data[i] = np.pad(data, (0, max_len - len(data)), 'constant', constant_values=0)

# Perform PCA with the peak data.
pca = PCA(n_components=3)  # reduce to three dimensions
peak_data = np.array(peak_data)
peak_data = peak_data.reshape(len(peak_data), -1)  # convert to 2D array
pca_result = pca.fit_transform(peak_data)

# Visualize PCA results with a 3D plot.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2])

# Display the file name next to each point.
for i, file_path in enumerate(file_paths):
    ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], file_path.split('/')[-1])

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA result for XRD spectra')
plt.show()

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3Dfrom sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Use KMeans to cluster the data.
n_clusters = 3  # Set the number of clusters. Here we use 4 as an example.
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_result)

# Get the cluster labels for each data point.
cluster_labels = kmeans.labels_

# Get a list of indices for data points belonging to each cluster.
clusters = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]

# Print the file names belonging to each cluster.
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for index in cluster:
        print(file_paths[index])
    print("\n")

# Reflect the cluster labels back in the visualization.
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=cluster_labels)

# Label each point with the file name.
for i, file_path in enumerate(file_paths):
    ax.text(pca_result[i, 0], pca_result[i, 1], pca_result[i, 2], file_path.split('/')[-1])

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA result for XRD spectra with clustering')
plt.legend(*scatter.legend_elements())
plt.show()