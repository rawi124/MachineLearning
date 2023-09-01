import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import pandas as pd
import random as rn

def generate_2d_data():
    """
    Generate synthetic dataset
    PARAM
        None
    RETURN
        v_x : array of 2d data points
    """
    v_x, _ = ds.make_blobs(n_samples=500, centers=3, n_features=2, random_state=30)
    return v_x

def dist_euc(a, b):
    """
    PARAM
        a : first point
        b : second point
    RETURN
        return euclidien distance between two points
    """
    distance = (abs(a[0]-b[0])**2 +abs(a[1]-b[1])**2)**0.5
    return distance
    
def dist_manthan(a, b):
    """
    PARAM
        a : first point
        b : second point
    RETURN
        return manathan distance between two points
    """
    
    return abs(b[0]-a[0])+abs(b[1]-a[1])
	
def init_centroid(k, v_x):
    """
    PARAM
        k : number of cluster
        v_x : data points
    RETURN
        v_c : centroids positions
    """
    #methode à banir 
    """
    maximum = np.max(v_x, axis = 0)
    minimum = np.min(v_x, axis = 0)
    l=[]
    i = 0
    while(i < k):
    
    	x = np.random.uniform(minimum[0], maximum[0])
    	y = np.random.uniform(minimum[1], maximum[1])
    	l.append([x, y])
    	i += 1
    return np.array(l)
    """
    l_centroide = []
    i = 0
    while(i < k):
    	alea = rn.randint(0, len(v_x)-1)
    	l_centroide.append(v_x[alea])
    	i +=  1
    return np.array(l_centroide)
    
def assign_cluster(k, v_x, v_c):
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
    i, j = 0, 0
    n = len(v_x)
    m_clusters = [[] * n for j in range(k)]
    #initialiser une matrice de dimensions (k, n)
    #chaque ligne i de la matrice contiendra les points les plus proches du centroide v_c[i]
    while(j < n):
    	dis_min = np.inf
    	i = 0
    	centroide = i
    	while(i < k):
    		dis_euclidienne = dist_euc(v_c[i], v_x[j])
    		if (dis_euclidienne < dis_min) :
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
	while( i < k): 
		centr = []
		j = 0
		s_x, s_y = 0, 0
		while (j < len(data[i])):
			s_x += data[i][j][0]
			s_y += data[i][j][1]
			j += 1
		centr.append(s_x/len(data[i]))
		centr.append(s_y/len(data[i]))
		l_centr.append(centr)
		i += 1
	l_centr = np.array(l_centr)
	return l_centr
	
#diffence between the two functions compute_centroids_conv and compute_centroids_iterations 
#is that the first one stops iterations when distance between two centroids is == to conv
#and the second stops working when nb iterations is == iterations 
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
    v_new_c = new_centroid(clusters, k)#nouveaux centroid    
    while(dist_euc(v_new_c[0], v_c[0]) > conv):
    	v_c = v_new_c
    	clusters = assign_cluster(k, v_x, v_c)
    	v_new_c = new_centroid(clusters, k)
    	nb_iterations += 1
    print("le nombre d'iterations pour arriver à convergence est egale à ",nb_iterations)
    return v_new_c#.flatten() pour transformer en tableau
    
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
    v_new_c = new_centroid(clusters, k)#nouveaux centroid      
    while(i<nb_iterations):
    	v_c = v_new_c
    	clusters = assign_cluster(k, v_x, v_c)
    	v_new_c = new_centroid(clusters, k)
    	i += 1
    return v_new_c#.flatten() pour transformer en tableau
        
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
    fig, ax = plt.subplots()
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