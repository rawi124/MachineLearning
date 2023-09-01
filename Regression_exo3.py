import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Regression_exo2 import gradient_descent, load_house_data, display_1d, norm_features, cost_func

EPS = 10e-10

    
def polyn_regression(x, y, nb_iter=5, lr= 0.2):
    # Initialisation des paramètres
    save_loss = list()
    save_params = list()
    params = (np.random.rand(x.shape[1]+1)-0.5) * [10000, 10000, 10000, 10000]
    x = norm_features(x) #normalisation des données

    for i in range(nb_iter):
        save_params.append(params)
        y_pred = predict_1(x, params)
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



def predict_1(x, params):
	"""
	en utilisant une equation du type : F(x,\theta) = \theta^T x
	"""
	
	
	# Pour une donnée d'entrée x, retourne la valeur y prédit par le model.
	x_transpose = x.transpose()
	
	return params[0] + params[1]*x_transpose[0] + params[2]*(x_transpose[1]**2) + params[3]*(x_transpose[2]**3)  


def predict_2(x, params):
	"""
	en utilisant une equation avec des racines carrees et cubique
	"""
	
	# Pour une donnée d'entrée x, retourne la valeur y prédit par le model.
	x_transpose = x.transpose()
	
	return params[0] + params[1]*x_transpose[0] + params[2]*(pow(x_transpose[1], 1/2)) + params[3]*(pow(x_transpose[2], 1/3)) 



if __name__ == "__main__":
    x, y = load_house_data()
    polyn_regression(x, y)
