import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def load_iris_train_data():
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
    train_data = pd.read_csv("data/iris_train.csv", skiprows=1, header=None, names=col_names)
    train_data = shuffle(train_data)
    y = np.array(train_data)[:,-1]
    x = np.array(train_data)[:,:-1]
    return x, y

def load_iris_validation_data():
    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']
    val_data = pd.read_csv("data/iris_validation.csv", skiprows=1, header=None, names=col_names)
    val_data = shuffle(val_data)
    y = np.array(val_data)[:,-1]
    x = np.array(val_data)[:,:-1]
    return x, y

def display_dataset(x, y):
    l_y = list(y)
    print("Nombre d'element dans le dataset : " + str(len(l_y)))
    print("Proportion de chaque classe dans le dataset")
    print(l_y.count('Setosa'), l_y.count('Virginica'), l_y.count('Versicolor'))
    color = 1 * (y == 'Setosa')
    color += 2* (y == 'Versicolor')
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=color)
    plt.figure()
    plt.scatter(x[:, 2], x[:, 3], c=color)
    plt.show()

class Node():
    def __init__(self, index_feature=None, threshold=None, tree_left=None, tree_right=None, info_gain=None, class_value=None):
        ''' Constructor '''
        # for decision node
        self.idx_feature = index_feature
        self.threshold = threshold

        self.left = tree_left    # left child
        self.right = tree_right  # right child
        self.info_gain = info_gain

        # for leaf node        nb_s, nb_vi, nb_ve = 0, 0, 0
        self.class_value = class_value


class DecisionTree():
    def __init__(self, min_samples_split=3, max_depth=2):
        ''' Constructor '''

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth


    def split(self, dataset, idx_feature, threshold):
        ''' function to split the data '''
        x = dataset[0]
        y = dataset[1]
        x_left = x[x[:, idx_feature] <= threshold]
        y_left = y[x[:, idx_feature] <= threshold]

        x_right = x[x[:, idx_feature] > threshold]
        y_right = y[x[:, idx_feature] <= threshold]

        dataset_left = [x_left, y_left]
        dataset_right = [x_right, y_right]
        return dataset_left, dataset_right


    def compute_dominante_label(self, y):
        ''' compute the dominante in y '''
        y = list(y)
        return max(y, key=y.count)


    def information_gain(self, y_parent, y_left_child, y_rigth_child):
        ''' function to compute information gain '''
        # Calcule le gain d'information par rapport au noeud parent et retourne ce gain
        gain = self.gini_index(y_parent)
        gain -= len(y_parent) * self.gini_index(y_left_child) / len(y_parent)
        gain -= len(y_parent) * self.gini_index(y_rigth_child) / len(y_parent)
        return gain


    def gini_index(self, y):
        ''' function to compute gini index '''
        l_y = list(y)
        n = len(l_y)
        # Pour chaque classe, on va mesurer sa proportion dans le dataset (sa probabilité)
        # Puis, on va sommer le carré de ces probabilités et retourner cette valeur
        return 1-((l_y.count('Setosa')/n)**2 + (l_y.count('Virginica')/n)**2+ (l_y.count('Versicolor')/n)**2)


    def print_tree(self, tree=None, level=0):
        ''' display the decision tree '''

        if tree == None:
            tree = self.root

        if tree.class_value is not None:
            print("%sLEAF: %s"%('\t'*level, tree.class_value))

        else:
            print("%sDECISION x%d <= %f"%('\t'*level, tree.idx_feature, tree.threshold))
            if tree.left is not None:
                self.print_tree(tree.left, level+1)
            if tree.right is not None:
                self.print_tree(tree.right, level+1)


    def build_tree(self, dataset, curr_depth=0):
        x, y = dataset[0], dataset[1]
        nb_samples, nb_features = np.shape(x)

        # Si la profondeur maximum n'est pas atteinte
            # Trouver la meilleure séparation
            # Si la séparation a apporté un gain d'information
                #left_subtree = self.build_tree(..., curr_depth+1)
                #right_subtree = self.build_tree(..., curr_depth+1)

                # On retourne un noeud avec l'indice de la feature utilisée et le seuil,
                # le sous-arbre gauche et le sous-arbre droit ainsi que l'information_gain

        # Si profondeur maximale atteinte ou que le noeud est pur
        # On calcule le label dominant
        # retourne le noeud (feuille) avec le label dominant


    def get_best_split(self, dataset):
        ''' function to find the best split '''
        # Dans cette fonction, on cherche la séparation qui permet d'avoir
        # la séparation de donnée la plus homogène (soit un gain d'information maximal)

        # Pour garder toutes les informations de cette séparation
        # nous allons utiliser un dictionnaire
        # l'indice de la feature à utiliser, la valeur du seuil
        # le sous-dataset gauche et droit et le gain d'information
        best_split = {}
        
        val_possibles = list((np.array(dataset[0])).transpose())
        l_features_distincts = []
        for val in val_possibles :
        	l_features_distincts.append(set(val))

        nb_features = len(l_features_distincts)
        meilleur_gain = 0
        i = 0
        while i < nb_features : 
        	for feature in l_features_distincts[i] :
        		dataset_left, dataset_right = self.split( dataset, i, feature)
        		if (len(dataset_left)>0 and len(dataset_right)>0):
        			#estce que dans dataset[0] il ya les x et dans data[1] ya label 
        			info_gain = self.information_gain( dataset[1], dataset_left[1], dataset_right[1])
        			if info_gain > meilleur_gain :
        				best = i, feature, dataset_left, dataset_right, info_gain
        	i += 1
        		
		
        # Pour toutes les features
            # Pour toutes les valeurs possibles
                # Séparer les données selon cette valeur
                # Si le dataset gauche et le dataset droit ne sont pas vides
                    # Calcule du gain d'information de cette séparation

        # Retourner le dictionnaire best split
        best_split[(best[0], best[1])] = best[2], best[3], best[4]
        return best_split

    def fit(self, x, y):
        ''' function to train the tree '''
        dataset = np.concatenate((x, y), axis=1)
        # Appeler la constructeur de l'arbre de décision et l'assigner à la racine

    def predict_one(self, x):
        pass

    def predict_all(self, x):
        pass

if __name__ == "__main__":
    n  = load_iris_train_data()
    #print(x_train)

    x_val, y_val  = load_iris_validation_data()

    # print("Display dataset")
    #display_dataset(x_train, y_train)
    # display_dataset(x_val, y_val)

    # Afficher l'arbre
    # Faire un abre manuellement
    fake_tree = DecisionTree()
    feuille1 = Node(class_value='Versicolor')
    feuille2 = Node(class_value='Setosa')
    feuille3 = Node(class_value='Versicolor')
    node1 = Node(1, 1.5, feuille1, feuille2, 0.5)
    node2 = Node(2, 2, node1, feuille3, 0.6)
    fake_tree.root = node1

    fake_tree.print_tree()
    c = DecisionTree()

    best = c.get_best_split(n)
    print(best)
    # Fit le modèle

    # Test le modèle
