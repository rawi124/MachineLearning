# -*- coding: utf-8 -*-
"""
implémente l'algorithme de classification K-means
"""
import random as rn
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds


def generate_2d_data():
    """
    Génération des données
    """
    v_x, _ = ds.make_blobs(n_samples=500, centers=3,
                           n_features=2, random_state=30)
    return v_x


import numpy as np

def dist_euc(point_1, point_2):
    """
    Retourne la distance euclidienne entre deux points de n'importe quelle dimension.
    """
    distance_carree = 0
    for i in range(len(point_1)):
        distance_carree += (point_1[i] - point_2[i])**2
    distance = np.sqrt(distance_carree)
    return distance

def dist_manthan(point_1, point_2):
    """
    Retourne la distance de Manhattan entre deux points
    """
    return abs(point_2[0]-point_1[0]) + abs(point_2[1]-point_1[1])


def init_centroid(k, v_x):
    """
    PARAM
        k : nb clusters
        v_x : jeu de données
    RETURN
        v_c : position des centroides
    """
    l_centroide = []
    i = 0
    while i < k:
        alea = rn.randint(0, len(v_x)-1)
        l_centroide.append(v_x[alea])
        i += 1
    return np.array(l_centroide)


def assign_cluster(k, v_x, v_c):
    """
    associe un cluster pour chaque point
    PARAM
        k : nb cluster
        v_x : donnees
        v_c : tableau de  centroids
    RETURN
        matrice (len = k)
        pour chaque element du v_c, on associe un cluster
    """
    i, j = 0, 0
    n = len(v_x)
    m_clusters = [[] * n for j in range(k)]
    while j < n:
        dis_min = np.inf
        i = 0
        centroide = i
        while i < k:
            dis_euclidienne = dist_euc(v_c[i], v_x[j])
            if dis_euclidienne < dis_min:
                dis_min = dis_euclidienne
                centroide = i
            i += 1
        m_clusters[centroide].append(list(v_x[j]))
        j += 1
    return m_clusters


def new_centroid(data, k):
    """
    calcule les nouveaux centroides et ceci en faisant la moyenne des points dans chaque cluster  

    """
    l_centr = []
    i = 0
    while i < k:
        centr = []
        j = 0
        s_x, s_y = 0, 0
        while j < len(data[i]):
            s_x += data[i][j][0]
            s_y += data[i][j][1]
            j += 1
        centr.append(s_x/len(data[i]))
        centr.append(s_y/len(data[i]))
        l_centr.append(centr)
        i += 1
    l_centr = np.array(l_centr)
    return l_centr


def compute_centroids_conv(k, v_x, conv):
    """
    compute new centroid positions
    PARAM
        k : number of cluster
        v_x : array of data points
        conv : when the difference between centoids == conv the fonction stops
    RETURN
        v_new_c : new centroid positions
    """
    nb_iterations = 0
    v_c = init_centroid(k, v_x)
    clusters = assign_cluster(k, v_x, v_c)
    v_new_c = new_centroid(clusters, k)  # nouveaux centroid
    while dist_euc(v_new_c[0], v_c[0]) > conv:
        v_c = v_new_c
        clusters = assign_cluster(k, v_x, v_c)
        v_new_c = new_centroid(clusters, k)
        nb_iterations += 1
    print("le nombre d'iterations pour arriver à convergence est egale à ", nb_iterations)
    return v_new_c  # .flatten() pour transformer en tableau


def compute_centroids_iterations(k, v_x, v_c, nb_iterations):
    """
    compute new centroid positions
    PARAM
    k : number of cluster
    v_x : array of data points
    nb_iterations : nb of iterations before stopping 
    RETURN
    v_new_c : new centroid positions
    """
    i = 0
    v_c = init_centroid(k, v_x)
    clusters = assign_cluster(k, v_x, v_c)
    v_new_c = new_centroid(clusters, k)  # nouveaux centroid
    while i < nb_iterations:
        v_c = v_new_c
        clusters = assign_cluster(k, v_x, v_c)
        v_new_c = new_centroid(clusters, k)
        i += 1
    return v_new_c  # .flatten() pour transformer en tableau


def display(k, v_c, clusters):
    """
    Display dataset
    PARAM
        v_x : array of data points
        v_c : array of centroids
    RETURN
        None
    displays ccentroides and clusters with different colors
    """
    colors = ['#13EAC9', '#FF00FF', '#9A0EEA']
    ax = plt.subplots()[1]
    for i in range(k):
        points = np.array(clusters[i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i])
        ax.scatter(v_c[:, 0], v_c[:, 1], marker='+', s=200, c='#050505')
    plt.show()


def k_means(k, conv):
    """
    PARAM
        k : number of cluster
    """
    v_x = generate_2d_data()
    v_c = compute_centroids_conv(k, v_x, conv)
    clusters = assign_cluster(k, v_x, v_c)
    display(k, v_c, clusters)


if __name__ == "__main__":
    k_means(3, 0.01)
