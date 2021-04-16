from __future__ import print_function

import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import misc
from struct import unpack

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def plot_mean_image(X, log):
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(meanrow,(28,28)), cmap=plt.cm.binary)
    plt.title('Mean image of ' + log)
    plt.show()

def get_labeled_data(imagefile, labelfile):
    """
    Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
    Adapted from: https://martin-thoma.com/classify-mnist-with-pybrain/
    """
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    X = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for id in range(rows * cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            X[i][id] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (X, y)


def my_clustering_mnist(X, y, n_clusters):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    
    fig, axs = plt.subplots(1,len(centers),figsize=(10,10))
    
    for ax, ci in zip(axs, centers):
        #axs.imshow(np.reshape(ci,(28,28)))
        #axs.show()
        Vec = np.reshape(ci,(28,28))
        ax.imshow(Vec)
    plt.show()
    C = y_kmeans
    
    ari_score = metrics.adjusted_rand_score(y, C)
    mri_score = metrics.mutual_info_score(y,C)
    v_measure_score = metrics.v_measure_score(y,C)
    silhouette_avg = metrics.silhouette_score(X, C, metric='euclidean')
    
    return [ari_score, mri_score, v_measure_score, silhouette_avg]

def main():
    # Load the dataset
    fname_img = 't10k-images.idx3-ubyte'
    fname_lbl = 't10k-labels.idx1-ubyte'
    [X, y]=get_labeled_data(fname_img, fname_lbl)

    # Plot the mean image
    plot_mean_image(X, 'all images')


    # Clustering
    range_n_clusters = [8, 9, 10, 11, 12]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)
    ari_array = []
    mri_array = []
    v_measure_array = []
    silhouette_array = []

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering_mnist(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])
        ari_array.append(ari_score[i])
        mri_array.append(mri_score[i])
        v_measure_array.append(v_measure_score[i])
        silhouette_array.append(silhouette_avg[i])

    def n_clusters(range_n_clusters):
        plt.figure(figsize=(10,10))
        plt.plot(range_n_clusters, ari_array, marker='o', color='blue', linewidth=2, label="ari_score")
        plt.plot(range_n_clusters, v_measure_array, marker='s', color='black', linewidth=2,  label="v_measure_score")
        plt.plot(range_n_clusters, silhouette_array, marker='X', color='brown', linewidth=2, label="silhouette_avg")
        plt.plot(range_n_clusters, mri_array, marker='*', color='red', linewidth=2, label="mri_score")
        plt.xlabel("No. of cluster")
        plt.ylabel("Score")
        plt.legend()
        plt.show()
     
    n_clusters(range_n_clusters)
    
if __name__ == '__main__':
    main()