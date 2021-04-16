from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics


def create_dataset():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=[0.5, 0.5, 1, 1, 1],random_state=3320)
    plt.title("data", fontsize='medium')
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    return [X, y]

def my_clustering(X, y, n_clusters):

    m=X.shape[0]
    n=X.shape[1]
    K = n_clusters
    Center=np.array([]).reshape(n,0) 
    for i in range(K):
        rand=np.random.randint(0,m-1)
        Center=np.c_[Center,X[rand]]
    Output={}
    for i in range(m):
        EDS=np.array([]).reshape(m,0)
        for k in range(K):
            tempDist=np.sum((X-Center[:,k])**2,axis=1)
            EDS=np.c_[EDS,tempDist]
        C=np.argmin(EDS,axis=1)+1
        Y={}
        for k in range(K):
            Y[k+1]=np.array([]).reshape(2,0)
        for i in range(m):
            Y[C[i]]=np.c_[Y[C[i]],X[i]] 
        for k in range(K):
            Y[k+1]=Y[k+1].T
        for k in range(K):
            Center[:,k]=np.mean(Y[k+1],axis=0)
        Output=Y
    for k in range(K):
        print("The centers are: ",Center[:,k])
    for key, value in Y.items():
        print("The size for cluster", key," is: ", len(value))

    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1])
    plt.scatter(Center[0,:],Center[1,:],s=200,c='yellow',label='Centroids')
    plt.show()
    C = [j - 1 for j in C]
    ari_score = metrics.adjusted_rand_score(y, C)
    mri_score = metrics.mutual_info_score(y,C)
    v_measure_score = metrics.v_measure_score(y,C)
    silhouette_avg = metrics.silhouette_score(X, C, metric='euclidean')
    return[ari_score, mri_score, v_measure_score, silhouette_avg]

def main():
    X, y = create_dataset()
    range_n_clusters = [2, 3, 4, 5, 6, 7]
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
        # Implement the k-means by yourself in the function my_clustering
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
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

