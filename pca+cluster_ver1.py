import pandas as pd
import glob
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read all the csv files.
file_paths = glob.glob('xrd_dirac/*.csv')

# Read the spectrum data from each csv file.
xrd_dataframes = [pd.read_csv(file_path) for file_path in file_paths]

# Find peaks in all XRD patterns.
peak_data = []
for df in xrd_dataframes:
    y_values = df['y'].values
    peaks, _ = find_peaks(y_values, height=0)  # height=0: find all peaks
    peak_data.append(y_values[peaks])

# Find the spectrum with the maximum number of peaks
max_len = max(len(data) for data in peak_data)

# Pad the peak data of all the spectra to have the same length
for i, data in enumerate(peak_data):
    if len(data) < max_len:
        peak_data[i] = np.pad(data, (0, max_len - len(data)), 'constant', constant_values=0)

# Perform PCA using the found peak data.
pca = PCA(n_components=2)  # reduce to 2 dimensions
peak_data = np.array(peak_data)
peak_data = peak_data.reshape(len(peak_data), -1)  # convert to 2D array
pca_result = pca.fit_transform(peak_data)

# Visualize the PCA result in a 2D plot.
plt.figure(figsize=(10, 7))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# Display the file name next to each point.
for i, file_path in enumerate(file_paths):
    plt.text(pca_result[i, 0], pca_result[i, 1], file_path.split('/')[-1])

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA result for XRD spectra')
plt.show()


# Clustering is performed using KMeans.
n_clusters = 3  # Set the number of clusters. In this example, 3 is used.
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_result)

# Obtain the cluster labels for each data point.
cluster_labels = kmeans.labels_

# Obtain the indices of data points that belong to each cluster as a list.
clusters = [np.where(cluster_labels == i)[0] for i in range(n_clusters)]

# Print the file names that belong to each cluster.
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for index in cluster:
        print(file_paths[index])
    print("\n")

# Reflect the cluster labels back to the visualization.
plt.figure(figsize=(10, 7))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels)

# Display the file name next to each point.
for i, file_path in enumerate(file_paths):
    plt.text(pca_result[i, 0], pca_result[i, 1], file_path.split('/')[-1])

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA result for XRD spectra with clustering')
plt.legend(*scatter.legend_elements())
plt.show()
