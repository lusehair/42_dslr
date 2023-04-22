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

		labels = data_train.select_dtypes(include=['int64', 'float64']).columns
		# print(labels)
		x_train = data_train[labels]
		y_train = data_train[['Hogwarts House']].values

		# 2. numerize y labels
		Y_tr = num_houses(y_train)

		X = {}
		for feature in labels:
			X[feature] = x_train[[feature]].values

		# ax, fig = plt.subplots(len(labels), len(labels), figsize=(25, 25))
		# i = 0
		# j = 0

		sns.pairplot(data=x_train, hue=Y_tr, diag_kind='kde')
		# sns.histplot(X[feature][(Y_tr == 1)], bins=20, color='r', ax=fig[i][j], kde=True)
		# for f1 in labels:
		# 	# ax.tight_layout(pad=5.0)
		# 	for f2 in labels:
		# 		print(x_train[(Y_tr == 1)])
		# 		sns.scatterplot(data=x_train[[feature]][(Y_tr == 1)], x=f1, y=f2, ax=fig[i][j])
		# 		# fig[i][j].scatter(X[f1][(Y_tr == 1)], X[f2][(Y_tr == 1)], marker='o', c='r', label='Gryffindor')
		# 		# fig[i][j].scatter(X[f1][(Y_tr == 2)], X[f2][(Y_tr == 2)], marker='o', c='y', label='Hufflepuff')
		# 		# fig[i][j].scatter(X[f1][(Y_tr == 3)], X[f2][(Y_tr == 3)], marker='o', c='b', label='Ravenclaw')
		# 		# fig[i][j].scatter(X[f1][(Y_tr == 4)], X[f2][(Y_tr == 4)], marker='o', c='g', label='Slytherin')

		# 		if i == len(labels) - 1:
		# 			fig[i][j].set_xlabel(f2)
		# 		if j == 0:
		# 			fig[i][j].set_ylabel(f1)
		# 		# fig[i][j].legend()
		# 		# fig[i][j].grid()
		# 		j += 1
		# 		if j == len(labels):
		# 			i += 1
		# 			j = 0

		plt.show()

	except Exception as e:
		print(e)