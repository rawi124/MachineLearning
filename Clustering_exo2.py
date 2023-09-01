"""
K-means avec des donnees en 3D
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds

from Clustering_exo1 import init_centroid, assign_cluster, dist_euc


def load_data():
    """
    Récupération des données en 3D
    """
    iris_data = ds.load_iris()
    m_x = iris_data.data
    return m_x[:, 1:]


def new_centroid_3d(clust, k):
    """
    calcule les nouveaux centroides avec 3 features(x, y, z)
    """
    l_centr = []
    i = 0
    while i < k:
        centr = []
        j = 0
        s_x, s_y, s_z = 0, 0, 0
        while j < len(clust[i]):
            s_x += clust[i][j][0]
            s_y += clust[i][j][1]
            s_z += clust[i][j][2]
            j += 1
        if len(clust[i]) != 0:
            centr.append(s_x/len(clust[i]))
            centr.append(s_y/len(clust[i]))
            centr.append(s_z/len(clust[i]))
            l_centr.append(centr)
        i += 1
    l_centr = np.array(l_centr)
    return l_centr


def compute_centroids_conv_3d(k, v_x, conv):
    """
    retourne  v_new_c : nouveaux centroides
    """
    nb_iterations = 0
    v_c = init_centroid(k, v_x)
    clusters = assign_cluster(k, v_x, v_c)
    v_new_c = new_centroid_3d(clusters, k)  # nouveaux centroid

    while dist_euc(v_new_c[0], v_c[0] > conv):
        v_c = v_new_c
        clusters = assign_cluster(k, v_x, v_c)
        v_new_c = new_centroid_3d(clusters, k)
        nb_iterations += 1
    print("le nombre d'iterations pour converger est egale à ", nb_iterations)

    return v_new_c  # .flatten() pour transformer en tableau


def scatterplot_3d(k, v_c, clusters):
    """
    affichage en 3d des clusters distingués par differentes couleurs
    """
    colors = ['#13EAC9', '#FF00FF', '#9A0EEA']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(k):
        points = np.array(clusters[i])
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors[i])
        ax.scatter(v_c[:, 0], v_c[:, 1], v_c[:, 2],
                   marker='+', s=200, c='#050505')
    plt.show()


def k_means_3d(k, conv):
    """
    fais appel aux differentes etapes de l'algorithme K-means
    """
    v_x = load_data()
    v_c = compute_centroids_conv_3d(k, v_x, conv)
    clusters = assign_cluster(k, v_x, v_c)
    scatterplot_3d(k, v_c, clusters)


if __name__ == "__main__":
    k_means_3d(3, 0.001)
