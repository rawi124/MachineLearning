import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_blobs
from Classification_ex1 import display_classification, dominant_label, nearest_neighbor, accuracy_score, precision_score, recall_score

def load_penguins_data():
	"""
	Generate synthetic dataset
	PARAM
		None
	RETURN
		m_x : dataset
		v_label : vector of label (0 or 1)
	"""
	m_raw = pd.read_csv("data/data_penguins.csv")
	m_x, v_label = m_raw[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]], m_raw["species"]
	return np.array(m_x), np.array(v_label)


def stat_dataset(m_x):
	"""
	Print statistics about dataset
	PARAM
		m_x : 
	RETURN 
		None
	"""
	print("======================analyse des donnees=====================================")
	m_raw = pd.read_csv("data/data_penguins.csv")
	m_raw.head()
	m_raw.info()
	m_raw.describe()
	sns.countplot(x=m_raw['species'])
	plt.show()
	sns.scatterplot(x='bill_length_mm', y='bill_depth_mm',data=m_raw,hue='species')
	plt.show()

def calculate_echantillon(y):
	
	c1, c2, c3 = 0, 0, 0
	n = len(y)
	dici = {}
	for el in y :
		if el == 'Gentoo':
			c1 += 1
		elif el == 'Chinstrap' :
			c2 += 1
		else :
			c3 += 1
	dici['Gentoo'] = (c1, c1/n) 
	dici['Chinstrap'] = (c2, c2/n)
	dici['Adelie'] = (c2, c2/n)
	moy_pondere = (1*c1+2*c2+3*c3)/n
	ecart_type = c1*(1-moy_pondere)**2+c2*(2-moy_pondere)**2+c3*(3-moy_pondere)
	#print(dici)
	return dici

def decoupe(x, y):
	"""
	decoupe le dataset en : train set, valid set et test set
	retourne un tuple contenant ces differnts ensembles 
	avec pour le trainset la majorite des elements : 222, valid_test : 55 et test_set 56
	"""
	n = len(x)

	len_train = 2*n//3
	x_train_set, x_valid_set, x_test_set = [], [], []
	y_train_set, y_valid_set, y_test_set = [], [], []
	
	x_train_set = x[0:len_train]
	y_train_set = y[0:len_train]
	
	len_valid = int(len_train+n/6)
	x_valid_set = x[len_train:len_valid]
	y_valid_set = y[len_train:len_valid]
	
	x_test_set = x[len_valid:n]
	y_test_set = y[len_valid:n]
	
	#return (train_set, valid_set, test_set)
	return x_train_set,  x_test_set,  y_train_set,  y_test_set
	#print(len(y_train_set), len(y_valid_set), len(y_test_set))
	#print(len(train_set),len(valid_set), len(test_set))

	
def effacer_espece(y):
	"""
	efface l espece adeli de y
	puisque auparavant il existait 119 espece de type Gento et 136 d Adelie et Chinstrap
	on a fusionné les deux especes adelie et christoph
	pour obtenir deux especes 119 et 136
	1 : Adelie + Chinstrap
	0 : Gento 
	"""
	new_y = []
	for el in y :
		if el == 'Gentoo':
			new_y.append(0)
		else :
			new_y.append(1)
	return new_y 
	
def knn(train_data, v_label, test_data, k=1):
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
	#print(y_pred)
	return y_pred 

def matrice_confusion(y_true, y_pred):
	return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred)


if __name__ == "__main__":
	x, y = load_penguins_data()
	#calculate_echantillon(y)
	#decoupe(x, y)
	y = effacer_espece(y)
	train_data, test_data,  y_train,  y_test_set = decoupe(x, y)
	y_pred = knn(train_data, y, test_data, k=3)
	#print(len(y), len(y_pred))
	print("matrice de confusion \n\n\n", matrice_confusion(y, y_pred))
	
	#print(x, "\n\n\n", y)
	#knn(x, y)
	
