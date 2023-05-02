import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Standard_Scaler
import pickle
from TinyStatistician import TinyStatistician as TS
from copy import deepcopy

# print out whole arrays
np.set_printoptions(threshold=np.inf)

# disable false positive warnings
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

		if y.ndim == 2:
			y = y.reshape(-1,)
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
		# fig.legend()

	except Exception as e:
		print(e)

def mean_(x):
	try:
		assert isinstance(x, pd.Series), "argument must be a panda series or dataframe"
		column_list = x.tolist() 
		m = 0
		cnt = 0
		for i in range(len(column_list)):
			if str(column_list[i]) != 'nan':
				m += column_list[i]
				cnt += 1
		m /= cnt
		return m 

	except Exception as e:
		print(e)

def median_(x):
	try:
		assert isinstance(x, pd.Series), "argument must be a panda series or dataframe"
		x_ = pd.Series(deepcopy(x.to_dict()))
		x_.dropna()
		column_list = x_.tolist() 
		m = 0
		cnt = 0
		n = len(column_list)
		column_list.sort()
		if (n % 2 == 1):
			return float(column_list[int(n / 2)])
		else:
			return float((column_list[int(n / 2) - 1] + column_list[int(n / 2)]) / 2)


	except Exception as e:
		print(e)

def mean_xy(x, y):
	try:
		assert isinstance(x, pd.Series), "argument must be a panda series or dataframe"
		x_ = x.tolist()

		m = {}
		cnt = {}
		for h in range(1, 5):
			m[h] = 0
			cnt[h] = 0
		for i in range(len(x_)):
			if str(x_[i]) != 'nan':
				m[y[i][0]] += x_[i]
				cnt[y[i][0]] += 1
		for h in range(1, 5):
			m[h] /= cnt[h]
		return m 

	except Exception as e:
		print(e)

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
		# labels.remove('History of Magic')
		# labels.remove('Transfiguration')
		# labels.remove('Divination')
		# labels.remove('Muggle Studies')
		# labels.remove('Flying')
		# labels.remove('Astronomy')
		# labels.remove('Defense Against the Dark Arts')
		# labels.remove('Herbology')
		# labels.remove('Ancient Runes')
		# labels.remove('Charms')
		print(labels)

		# 2. numerize y labels
		y = data_train[['Hogwarts House']].values
		y_train = num_houses(y)

		# Replace NaN value by mean
		# mean_train = []
		# cnt = 0
		# for col in data_train[labels]:
		# 	m = mean_(data_train[col])
		# 	# m = median_(data_train[col])
		# 	mean_train.append(m)
		# 	data_train[col].fillna(m, inplace=True)
		# df_mean = pd.DataFrame(data=np.array(mean_train), columns=['Mean'])
		# df_mean.to_csv('mean_train.csv', index=False)
		x = data_train[labels]
		m = {}
		for col in x:
			m[col] = mean_xy(x[col], y_train)
		for col in x:
			for i in range(len(x)):
				if str(x[col].iloc[i]) == 'nan':
					x[col].iloc[i] = m[col][y_train[i][0]]
		x_train = x.values
		
		features = labels
		houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		colors = ['r','y','b','g']
		# 2. numerize y labels
		y_train = num_houses(y)

		# 3. Normalization
		# Minmax
		# my_Scaler = Minmax_Scaler()
		# my_Scaler.fit(x_train)
		# X_tr = my_Scaler.transform(x_train)

		# Zscore
		my_Scaler = Standard_Scaler()
		my_Scaler.fit(x_train)
		X_tr = my_Scaler.transform(x_train)


		# 4. Training
		# We are going to train 4 logistic regression classifiers to discriminate each class from the others
		models = {}
		y_ = {}
		fig = plt.figure()
		fig.suptitle("Loss over time")
		# ax = plt.axes()
		for i in range(1, 5):
			# 4.a relabel y labels
			y_[i] = relabel(y_train, i)

			# 4.b training
			models[i] = MyLR(np.ones((X_tr.shape[1] + 1, 1)), alpha=2e-3, max_iter=6000)
			x_step, loss_time = models[i].fit_(X_tr, y_[i])
			# print(models[i].theta)
			plt.plot(x_step, loss_time, label=houses[i-1], linewidth=2, c=colors[i-1])
		plt.grid()
		plt.xlabel('Iterations')
		plt.ylabel('Loss')
		plt.legend()
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
		ax, fig = plt.subplots(1, sum(range(len(labels))), figsize=(30, 10), constrained_layout = True)
		cnt = set()
		k = 0
		for i in range(len(labels)):
			for j in range(len(labels)):
				if i != j and ((i, j) not in cnt) and i < j:
					cnt.add((i, j))
					scatter_plot(fig[k], x_train[:, i], x_train[:, j], y_train.reshape(-1,), y_pred_tr.reshape(-1,), labels[i], labels[j])
					k += 1
		fig[k - 1].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
		plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n" \
			+ "Percentage of correct predictions for train data:  " +   str(round(100 * MyLR.score_(y_pred_tr, y_train), 1)) + "%\n" + "labels: " + str(labels))
		plt.show()

		# 8. Save models
		with open("models.pickle","wb") as f:
			pickle.dump(models, f)

	except Exception as e:
		print(e)
