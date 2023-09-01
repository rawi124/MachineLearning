import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

EPS = 10e-10

def load_house_data():
    raw_dataset = pd.read_csv("house_prices.csv")
    raw_dataset = raw_dataset[['GrLivArea', 'YearBuilt', 'TotRmsAbvGrd', 'SalePrice']].dropna()

    x = np.array(raw_dataset[['GrLivArea']])
    y = np.array(raw_dataset['SalePrice'])
    return x, y

def display_1d(x, y, params):
    x = x[:,0]

    val = np.linspace(min(x), max(x), 1000)
    pred = val * params[1] + params[0]
    plt.figure()
    plt.scatter(x, y, marker='+')
    plt.plot(val, pred, '-', c='orange')
    plt.show()

def norm_features(x):
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0)+EPS)
    return x

def linear_regression(x, y, nb_iter, lr):
	params = (np.random.rand(x.shape[1]+1)-0.5) * [10000, 1000]

	save_loss = list()
	save_params = list()
	x = norm_features(x)

	for i in range(nb_iter):
		y_pred = predict(x, params)#y predit selon les parametre O1 et O0
		fct_cout = cost_func(y_pred, y)#fonction de cout selon le y predit 
		save_params.append(params)
		save_loss.append(fct_cout[0])
		display_1d(x, y, params)
		params =  gradient_descent(x, y, y_pred, params, lr)#les nouveaux parametres pour l'iteration prochaine

	plt.figure()
	plt.plot(save_loss)
	plt.title("Loss function")
	
	plt.figure()
	plt.plot(save_params)
	plt.legend(['theta1','theta2'])
	plt.title("Parameters")
	plt.show()
	

def predict(x, params):
	"""
	predit la valeur de y pour x en entrée 
	ici x est la variable dont on va etudier sa relation avec y 
	"""
	
	y = []
	for el in x :
		y.append(params[0] + params[1] * el)
	return y


def cost_func(y_pred, y_gt):
	"""
	la fonction coût est une fonction mathématique qui mesure l’erreur que nous commettons en approximant les données.
	"""
	# A partir de la prediction fait par le modèle et la vérité terrain, calculez la fonction de coût
	fct_cout = 0
	n, i = len(y_pred), 0
	while i < n :
		fct_cout += (y_pred[i]-y_gt[i])**2
		i += 1
	return fct_cout/(2*n)
def gradient_descent(x, y_gt, y_pred, params, learning_rate):
	""""
	sert à renvoyer les parametres tetas pour x, y , ypred, paramas et le learning_rate
	"""
	# a l'aide de la prédiction, calculer le gradiant et mettez à jour les parametres
	O0, O1 = params[0], params[1]
	n, i = len(x), 0
	grad_0, grad_1 = 0,0
	while i < n :
		add = y_pred[i]-y_gt[i]
		grad_0 += add
		grad_1 += add*x[i]
		i += 1
	grad_0 /= n
	grad_1 /= n
	O0 = O0 - learning_rate * grad_0
	O1 = O1 -  learning_rate * grad_1

	return O0, O1


if __name__ == "__main__":
    x, y = load_house_data()
    linear_regression(x, y, 5, lr= 0.5)



