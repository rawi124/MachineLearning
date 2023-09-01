# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.metrics


def generate_trainset():
	"""
	Generate synthetic dataset for the trainset
	PARAM
		None
	RETURN
		m_x : array of 2d data points
		v_label : vector of label (0 or 1)
	"""
	m_x, v_label = ds.make_blobs(n_samples=400, centers=2, n_features=2, cluster_std=[3, 3], random_state=42)
	return m_x, v_label


def generate_testset():
	"""
	Generate synthetic dataset for the testset
	PARAM
		None
	RETURN
		m_x : array of 2d data points
		v_label : vector of label (0 or 1)
	"""
	m_x, v_label = ds.make_blobs(n_samples=200, centers=2, n_features=2, cluster_std=[3.2, 3.2], random_state=42)
	return m_x, v_label


def display_2D_data(m_x, v_label):
	"""
	Display dataset
	PARAM
		m_x : array of data points
		v_label : array of centroids
	RETURN
		None
	"""
	v_color = "orange" if v_label == 0 else "green";

	plt.figure()
	plt.scatter(m_x[:, 0], m_x[:, 1], c=v_label)
	plt.show()

def dist_euc(a, b):
	"""
	PARAM
	a : first point
	b : second point
	RETURN
	return euclidien distance between two points 
	"""
	i = 0
	distance = 0
	while i < len(a) :  
		distance += abs(a[i]-b[i])**2 
		i+= 1
	return distance**0.5
	

def nearest_neighbor(k, dataset, x) : 
	"""
	qui retourne les indices des K points(voisins) les plus proches de x
	"""
	indices = []
	dist = []
	voisins = []
	i = 0
	while(i<len(dataset)):
		dist.append([dist_euc(dataset[i],x),i]) #on calcule la distance de chaque points par rapport au point x
		i += 1
	dist = sorted(dist,) #on ordonne notre liste de distance
	indices = dist[1:k+1] #on récupere les indices des k plus proches voisins de x
	for v in indices: #pour recuperer que les indices
		voisins.append(v[1])
	#print(voisins)
	return voisins

def dominant_label(voisins, y) : 
	"""
	qui retourne le label majoritaire dans un set de données voisins (qui est renvoyé par la fonction precedente)
	"""
	labels = []
	s = 0
	for i in range(len(voisins)):
		labels.append(y[voisins[i]])
	for el in labels :
		s += el 
	s = s/len(voisins)
	if s > 0.5 :
		#si la moyenne est superieur à 0.5 alors la classe majoritaire est 1
		return 1
	return 0

def dominant_label_wnn(train_data, voisins, y, x):
	"""
	retourne le label majoritaire dans un set de donnes en utilisant
	une autre variante de calcul de distance 
	"""
	labels = []
	s_w = 0
	s = 0
	j = 0
	for i in range(len(voisins)):
		labels.append(y[voisins[i]])
	while(j<len(voisins)):
		w = (1/dist_euc(train_data[voisins[i]],x))
		s_w += w
		s += w * labels[j]
		j += 1
	s = s / s_w
	if s < 0.5:
		return 0
	else:
		return 1

def dwnn(train_data, v_label, test_data, k):
	"""
	variante du KNN : la Distance-weighted Nearest Neighbor
	"""
	y_pred = []
	for i in test_data:
		voisins = nearest_neighbor(k, train_data, i)
		label = dominant_label_wnn(train_data, voisins, v_label, i)
		y_pred.append(label)
	#display_classification(test_data,y_pred)
	return y_pred
			
	

def display_classification(dataset, y) : 
	"""
	affichage du dataset reparti en deux label
	"""
	for i in range(len(dataset)):
		if y[i] == 0: 
			plt.scatter(dataset[i,0], dataset[i,1], marker='+', s=100, color='red')
		else:
			plt.scatter(dataset[i,0], dataset[i,1], marker='o', color='blue')
	plt.title("Figure : répartition des données selon les deux classes {0,1}.")
	plt.show()


def accuracy_score(y_true, y_pred):
	"""
	calculer le nombre d instances bien classé entre y_true renvoyé par la premiere fonction de genration 
	et y_pred renvoyé par la fonction de prediction
	"""
	
	tp_tn = 0
	i = 0
	for el in y_pred :
		if el == y_true[i]:
			tp_tn += 1 
		i += 1
	#print((tp_tn/len(y_true))*100, sklearn.metrics.accuracy_score(y_true, y_pred))
	return (tp_tn/len(y_true))*100
	#return sk.metrics.accuracy_score(y_true, y_pred)

def precision_score(y_true, y_pred):
	tp_fp = 0
	tp = 0
	i = 0
	for el in y_pred :
		if el == y_true[i] and el == 0:
			tp += 1 
		i += 1
	for ell in y_true :
		if ell == 0 :
			tp_fp += 1 
	#print((tp/tp_fp)*100, sklearn.metrics.precision_score(y_true, y_pred))
	return (tp/tp_fp)*100

def recall_score(y_true, y_pred):
	tp_fn = 0
	tp = 0
	i = 0
	for el in y_pred :
		if el == y_true[i] and el == 0:
			tp += 1 
		i += 1
	for ell in y_pred :
		if ell == 0 :
			tp_fn += 1 
	#print((tp/tp_fn)*100, sklearn.metrics.recall_score(y_true, y_pred))
	return (tp/tp_fn)*100


def knn(train_data, v_label, test_data, k):
	y_pred_app = []
	y_pred = []
	#print(v_label)
	"""
	#APPRENTISSAGE
	for el in train_data :
		voisins_app = nearest_neighbor(k, train_data, el)
		y_pred_app.append(dominant_label(voisins_app, v_label))
	"""
	#TEST
	for el in test_data :
		voisins = nearest_neighbor(k, train_data, el)
		y_pred.append(dominant_label(voisins, v_label))
	#display_classification(test_data,y_pred)
	return y_pred 
		

if __name__ == "__main__":
	x, y = generate_trainset()
	x_test, y_test = generate_trainset()
	#display_classification(x, y)
	print("verification des valeurs ")
	l_k = [1, 3, 5, 7]
	for k in l_k :
		print("=====================pour K = ", k, "=================================")
		y_n = knn(x, y, x_test, k)	
		acc = accuracy_score(y, y_n)
		re = recall_score(y, y_n)
		pre = precision_score(y, y_n)
		print("accuray score ",acc,"recall score ", re, "precision score ",pre )
		print("Matrice de confusion: de verification")
		matrix = sklearn.metrics.confusion_matrix(y_test, y_n)
		print(matrix)
	
	#l = dwnn(x, y, x_test, 3)	
	#print(dwnn(x, y, x_test, 7))
	
	
