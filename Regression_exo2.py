import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Regression_exo1 import cost_func

EPS = 10e-10

def load_house_data():
    raw_dataset = pd.read_csv("house_prices.csv")
    raw_dataset = raw_dataset[['GrLivArea', 'YearBuilt', 'TotRmsAbvGrd', 'SalePrice']].dropna()

    x = np.array(raw_dataset[['GrLivArea', 'YearBuilt', 'TotRmsAbvGrd']])
    y = np.array(raw_dataset['SalePrice'])

    return x, y

def norm_features(x):
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0)+EPS)
    return x


def display_1d(x, y, params):
    x = x[:,0]

    val = np.linspace(min(x), max(x), 1000)
    pred = val * params[1] + params[0]
    plt.figure()
    plt.scatter(x, y, marker='+')
    plt.plot(val, pred, '-', c='orange')
    plt.show()
    
def linear_regression(x, y, nb_iter=7, lr= 0.5):
    # Initialisation des paramètres
    save_loss = list()
    save_params = list()
    params = (np.random.rand(x.shape[1]+1)-0.5) * [10000, 10000, 10000, 10000]
    x = norm_features(x) #normalisation des données

    for i in range(nb_iter):
        save_params.append(params)
        y_pred = predict(x, params)
        save_loss.append(cost_func(y_pred, y))
        params = gradient_descent(x, y, y_pred, params, lr)
        display_1d(x, y, params)

    # Affichage des valeurs de la fonction de coût et des paramêtres au cours du temps
    plt.figure()
    plt.plot(save_loss)
    plt.title("Loss function")

    plt.figure()
    plt.plot(save_params)
    plt.legend(['theta1','theta2','theta3','theta4'])
    plt.title("Parameters")
    plt.show() 



def predict(x, params):
	
	# Pour une donnée d'entrée x, retourne la valeur y prédit par le model.
	x_transpose = x.transpose()
	
	return params[0] + params[1]*x_transpose[0] + params[2]*x_transpose[1] + params[3]*x_transpose[2]  


def gradient_descent(x, y_gt, y_pred, params, learning_rate):
	# a l'aide de la prédiction, calculer le gradiant et mettez à jour les parametres
	O0, O1, O2, O3  = params[0], params[1], params[2],  params[3]
	n, i = len(x), 0
	x = x.transpose()
	grad_0,grad_1,grad_2,grad_3 = 0, 0, 0, 0
	while i < n :
		add = y_pred[i]-y_gt[i]
		grad_0 += add
		grad_1 += add*x[0][i]
		grad_2 += add*x[1][i]
		grad_3 += add*x[2][i]
		i += 1
	grad_0 /= n
	grad_1 /= n
	grad_2 /= n
	grad_3 /= n
	O0 = O0 -  learning_rate * grad_0
	O1 = O1 - learning_rate * grad_1
	O2 = O2 -  learning_rate * grad_2
	O3 = O3 - learning_rate * grad_3
	return O0, O1, O2, O3

if __name__ == "__main__":
    x, y = load_house_data()
    



    linear_regression(x, y)
