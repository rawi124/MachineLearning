import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import pandas as pd

from Clustering_exo1 import init_centroid, assign_cluster

def load_data():
	m_x = ds.load_iris().data
	return m_x[:,1:]

def dist_euc_3D(a, b):
    """
    PARAM
        a : first point
        b : second point
    RETURN
        return euclidien distance between two points(3D)
    """
    distance = (abs(a[0]-b[0])**2 +abs(a[1]-b[1])**2+abs(a[2]-b[2]))**0.5
    return distance


def new_centroid_3D(clust, k):
	"""
	calculate new array of centroids with 3 features(x, y, z)
	"""
	l_centr = []
	i = 0
	while( i < k): 
		centr = []
		j = 0
		s_x, s_y ,s_z = 0, 0, 0
		while (j < len(clust[i])):
			s_x += clust[i][j][0]
			s_y += clust[i][j][1]
			s_z += clust[i][j][2]
			j += 1
		if len(clust[i]) != 0 :
			centr.append(s_x/len(clust[i]))
			centr.append(s_y/len(clust[i]))
			centr.append(s_z/len(clust[i]))
			l_centr.append(centr)
		i += 1
	l_centr = np.array(l_centr)
	return l_centr
	
def compute_centroids_conv_3D(k, v_x, conv):
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
    v_new_c = new_centroid_3D(clusters, k)#nouveaux centroid 
       
    while(dist_euc_3D(v_new_c[0], v_c[0]) > conv):
    	v_c = v_new_c 
    	clusters = assign_cluster(k, v_x, v_c)
    	v_new_c  = new_centroid_3D(clusters, k)
    	nb_iterations += 1
    print("le nombre d'iterations pour converger est egale à ",nb_iterations)
   
    return v_new_c #.flatten() pour transformer en tableau

def scatterplot_3d(k, v_c, clusters):
	"""
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	ax.scatter(v_x[:,0], v_x[:,1], v_x[:,2])
	plt.show()
	"""
	colors = ['#13EAC9', '#FF00FF', '#9A0EEA']
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	#print(v_c)
	for i in range(k):
		points = np.array(clusters[i])
		ax.scatter(points[:, 0], points[:, 1],points[:, 2], c=colors[i])
		ax.scatter(v_c[:, 0], v_c[:, 1], v_c[:, 2], marker='+', s=200, c='#050505')
	plt.show()

def k_means_3d(k, conv):
	v_x = load_data()
	v_c = compute_centroids_conv_3D(k, v_x, conv)
	clusters = assign_cluster(k, v_x, v_c)
	scatterplot_3d(k, v_c, clusters)
	
if __name__ == "__main__":
	k_means_3d(3, 0.001)
