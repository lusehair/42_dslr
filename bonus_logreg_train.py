import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from bonus_my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Standard_Scaler
import pickle

# print out whole arrays
np.set_printoptions(threshold=np.inf)

# disable false positive warings
pd.options.mode.chained_assignment = None  # default='warn'


def num_houses(y):
	try:
		assert isinstance(
				y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		dict_ = {}
		dict_['Gryffindor'] = 1
		dict_['Hufflepuff'] = 2
		dict_['Ravenclaw'] = 3
		dict_['Slytherin'] = 4

		houses = {'Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin'}

		y_ = []
		for i in range(len(y)):
			assert y[i][0] in houses, "house name not recognized"
			y_.append(dict_[y[i][0]])
		return np.array(y_).reshape(-1,1)

	except Exception as e:
		print(e)
		return None


def label_houses(y):
	try:
		assert isinstance(
				y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		dict_ = {}
		dict_[1] = 'Gryffindor'
		dict_[2] = 'Hufflepuff'
		dict_[3] = 'Ravenclaw'
		dict_[4] = 'Slytherin'

		y_ = []
		for i in range(len(y)):
			assert y[i] in range(1, 5), "house numerized label must be either 1, 2, 3 or 4"
			y_.append(dict_[y[i]])
		return np.array(y_).reshape(-1,1)

	except Exception as e:
		print(e)
		return None


def fill_zeros(x):
	try:
		assert isinstance(
				x, pd.DataFrame), "arguments should be a dataframe"
		for feature in labels:
			# x[feature] = x[feature].replace(r'\s+', np.nan, regex=True).fillna(0)
			x[feature] = x[feature].fillna(0)
		return x

	except Exception as e:
		print(e)
		return None


def relabel(y, fav_label):
	try:
		assert isinstance(
				y, np.ndarray) and (y.ndim == 1 or y.ndim == 2), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(fav_label, int) and fav_label in {1, 2, 3, 4}, "2nd argument must be a int that is either 0, 1 ,2 or 3"
		# for yi in y:
		# 	print(yi)
		# 	print(type(yi))
		return(np.array([1 if yi[0]==fav_label else 0 for yi in y])).reshape(-1, 1)

	except Exception as e:
		print(e)
	

def scatter_plot(fig, x1, x2, y_test, y_pred, xlabel, ylabel):
	try:
		assert isinstance(
				x1, np.ndarray) and (x1.ndim == 1), "1st argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				x2, np.ndarray) and (x2.ndim == 1), "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				y_test, np.ndarray) and (y_test.ndim == 1), "3rd argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(
				y_pred, np.ndarray) and (y_pred.ndim == 1), "4th argument must be a numpy.ndarray, a vector of dimension m * 1"
		assert isinstance(xlabel, str) and isinstance(ylabel, str), "5th, 6th and 7th arguments must be strings"

		
		fig.scatter(x1[(y_test == 1)], x2[(y_test == 1)], s = 200, color='tab:pink', label="true values: Gryffindor")
		fig.scatter(x1[(y_test == 2)], x2[(y_test == 2)], s = 200, color='tab:gray', label="true values: Hufflepuff")
		fig.scatter(x1[(y_test == 3)], x2[(y_test == 3)], s = 200, color='y', label="true values: Ravenclaw")
		fig.scatter(x1[(y_test == 4)], x2[(y_test == 4)], s = 200, color='c', label="true values: Slytherin")
		fig.scatter(x1[(y_pred == 4)], x2[(y_pred == 4)], marker='x', color='b', label="predictions: Slytherin")
		fig.scatter(x1[(y_pred == 1)], x2[(y_pred == 1)], marker='x', color='tab:purple', label="predictions: Gryffindor")
		fig.scatter(x1[(y_pred == 2)], x2[(y_pred == 2)], marker='x', color='g', label="predictions: Hufflepuff")
		fig.scatter(x1[(y_pred == 3)], x2[(y_pred == 3)], marker='x', color='tab:brown', label="predictions: Ravenclaw")
		fig.set_xlabel(xlabel)
		fig.set_ylabel(ylabel)
		fig.grid()
		fig.legend()

	except Exception as e:
		print(e)


def mean_(column): 
	column_list = column.tolist() 
	mean = 0 
	for i in range(len(column_list)): 
		mean += column_list[i]
		mean = mean / len(column_list)
		return mean 

if __name__ == "__main__":
	try:
		# assert len(sys.argv) >= 2, "missing path"
		# path = sys.argv[1]

		# 1. Load data
		path = "datasets/dataset_train.csv"
		data_train = pd.read_csv(path)
		# path = "datasets/dataset_test.csv"
		# data_test = pd.read_csv(path)
		# labels = ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
		labels = list(data_train.select_dtypes(include=['int64', 'float64']).columns)
		labels.remove('Index')
		labels.remove('Arithmancy')
		labels.remove('Potions')
		labels.remove('Care of Magical Creatures')
		labels.remove('History of Magic')
		labels.remove('Transfiguration')
		labels.remove('Divination')
		labels.remove('Muggle Studies')
		#labels.remove('Flying')
		#labels.remove('Astronomy')
		labels.remove('Defense Against the Dark Arts')
		labels.remove('Herbology')
		#labels.remove('Ancient Runes')
		#labels.remove('Charms') 

		['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Transfiguration', 'Charms', 'Flying']

		# Replace NaN value by mean 
		for col in data_train[labels]:
			mean = mean_(data_train[col])
			data_train[col].fillna(mean, inplace=True)
		
		
		print(len(data_train[labels]))
		x = data_train[labels].dropna()
		
		print(len(x))

		x_train = x.values
		y = data_train[['Hogwarts House']].values
		features = labels
		houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

		# 2. numerize y labels
		y_train = num_houses(y)

		# 3. Normalization
		# Zscore
		my_Scaler = Standard_Scaler()
		my_Scaler.fit(x_train)
		X_tr = my_Scaler.transform(x_train)


		# 4. Training
		# We are going to train 4 logistic regression classifiers to discriminate each class from the others

		models = {}
		y_ = {}
		for i in range(1, 5):
			# 4.a relabel y labels
			y_[i] = relabel(y_train, i)

			# 4.b training
			models[i] = MyLR(np.ones((X_tr.shape[1] + 1, 1)), alpha=0.005, max_iter=150000, _batch_size=300)
			models[i].fit_(X_tr, y_[i])
			# print(models[i].theta)

		# 5. Predict for each example the class according to each classifiers and select the one with the highest output probability.

		y_pred_tr_ = np.array([])
		for i in range(1, 5):
			if y_pred_tr_.any():
				y_pred_tr_ = np.hstack((y_pred_tr_, models[i].predict_(X_tr)))
			else:
				y_pred_tr_ = models[i].predict_(X_tr)

		# 6. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set and compare it to the train set.

		y_pred_tr = np.argmax(y_pred_tr_, axis=1).reshape(-1,1) + 1
		print("fraction of correct predictions for train data:  ", MyLR.score_(y_pred_tr, y_train))
		
		# 7. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
		# _, fig = plt.subplots(1, sum(range(len(labels))), figsize=(30, 10))
		# print (sum(range(len(labels))))
		# cnt = set()
		# k = 0
		# for i in range(len(labels)):
		# 	for j in range(len(labels)):
		# 		if i != j and ((i, j) not in cnt) and i < j:
		# 			print(k, i, j)
		# 			cnt.add((i, j))
		# 			scatter_plot(fig[k], x_train[:, i], x_train[:, j], y_train.reshape(-1,), y_pred_tr.reshape(-1,), labels[i], labels[j])
		# 			k += 1

		# plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n" \
		# 	+ "fraction of correct predictions for train data:  " +   str(round(100 * MyLR.score_(y_pred_tr, y_train), 1)))
		plt.show()

		# 8. Save models
		with open("models.pickle","wb") as f:
			pickle.dump(models, f)

	except Exception as e:
		print(e)