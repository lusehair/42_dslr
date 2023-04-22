import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy
import seaborn as sns


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

if __name__ == "__main__":
	try:

		# 1. Load data
		path = "datasets/dataset_train.csv"
		data_train = pd.read_csv(path)

		labels = list(data_train.select_dtypes(include=['int64', 'float64']).columns)
		# print(labels)
		x_train = data_train[labels]
		y_train = data_train[['Hogwarts House']].values

		# 2. numerize y labels
		Y_tr = num_houses(y_train)

		# 3. create a distionary of features numpy arrays
		X = {}
		for feature in labels:
			X[feature] = x_train[[feature]].values

		# 4. create pairplot using Seaborn
		labels.remove('Index')
		sns.pairplot(data=data_train[labels + ['Hogwarts House']], hue='Hogwarts House', diag_kind='kde', plot_kws={'s': 2}, height=1.5)


		plt.show()

	except Exception as e:
		print(e)