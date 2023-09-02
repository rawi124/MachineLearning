"""
application du k-means sur des images
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image

import Clustering_exo2 as C


def load_image(path):
    """
    load an image
    PARAM
            path : path + filename of the image
    RETURN
            img : numpy array
    """
    img = image.imread(path)

    if img.ndim == 3:
        img = img[:, :, :3]
    return img


def assign_cluster_img(k, v_x, v_c):
    """
    assign a cluster for each datapoints
    PARAM
        k : number of cluster
        v_x : array of data points
        v_c : array of centroids
    RETURN
        matrice (len = k)
        foreach element of the matrice there is elements of one cluster
    """
    # cette fonction enregistre les positions des 
    #points des clusters a la difference de celle implementé dans
    # les autres fichiers.
    i, j = 0, 0
    n = len(v_x)
    # initialise une matrice de k elements chaque 
    #ligne  i contient les points du ieme cluster
    m_clusters = [[] * n for j in range(k)]
    while j < n:
        dis_min = np.inf
        i = 0
        centroide = i
        while i < k:
            dis_euclidienne = C.dist_euc_3D(v_c[i], v_x[j])
            if dis_euclidienne < dis_min:
                dis_min = dis_euclidienne
                centroide = i
            i += 1
        m_clusters[centroide].append(j)
        j += 1
    return m_clusters


def k_means_im(k, conv, v_x):
    """
    applique k_means a 3 features a une image
    retourne la matrice de l image modifiee avec les nouvelles couleurs 
    chaque point de la matrice sera modifie par le centroide de son cluster

    """
    l_c = []
    copie = v_x  # garder la matrice en copie pour pouvoir la modifier plus bas dans la fonction
    for el in v_x:
        for ell in el:
            l_c.append(ell)
    v_x = l_c
    v_c = C.compute_centroids_conv_3d(k, v_x, conv)
    clusters = assign_cluster_img(k, v_x, v_c)
    i = 0
    for el in clusters:
        for ell in el:
            l_c[ell] = v_c[i]
        i += 1
    i, x = 0, 0
    while i < len(copie):
        j = 0
        while j < len(copie[0]):
            copie[i][j] = l_c[x]
            x += 1
            j += 1
        i += 1
    return copie


def display_image(img):
    """
    PARAM
            img: numy array
    RETURN
            None
    """
    plt.imshow(img, aspect='auto', interpolation=None)
    plt.show()


if __name__ == "__main__":
    img = load_image('data/landscape_200px.png')
    img = k_means_im(15, 0.08, img)
    display_image(img)
