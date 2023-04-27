import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Standard_Scaler
from logreg_train import num_houses, label_houses, fill_zeros, relabel, scatter_plot
import pickle


# print out whole arrays
np.set_printoptions(threshold=np.inf)

# disable false positive warings
pd.options.mode.chained_assignment = None  # default='warn'

if __name__ == "__main__":
	try:
		# assert len(sys.argv) >= 2, "missing path"
		# path = sys.argv[1]

		# 1. Load data
		path = "datasets/dataset_train.csv"
		data_train = pd.read_csv(path)
		path = "datasets/dataset_test.csv"
		data_testX = pd.read_csv(path)
		path = "datasets/dataset_truth.csv"
		data_testY = pd.read_csv(path)
		# labels = ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']

		labels = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'Ancient Runes', 'Charms']
		labels = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']

		x_train = data_train[labels]
		x_train = x_train.dropna()
		x_train = x_train.values

		x = data_testX[labels]
		y = data_testY[['Hogwarts House']]
		tmp = [x, y]
		x = pd.concat(tmp, axis=1)

		x = x.dropna()
		y = x[['Hogwarts House']].values
		x = x[labels]
		x_test = x.values

		features = labels
		houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		
		# 2. numerize y labels
		y_test = num_houses(y)

		# 3. Normalization
		# Zscore
		my_Scaler = Standard_Scaler()
		my_Scaler.fit(x_train)
		# X_tr = my_Scaler.transform(x_train)
		X_te = my_Scaler.transform(x_test)

		# 5.load models
		# We are going to train 4 logistic regression classifiers to discriminate each class from the others
		with open("models.pickle", "rb") as f:
			models = pickle.load(f)

		# 5. Predict for each example the class according to each classifiers and select the one with the highest output probability.
		y_pred_ = np.array([])
		for i in range(1, 5):
			if y_pred_.any():
				y_pred_ = np.hstack((y_pred_, models[i].predict_(X_te)))
			else:
				y_pred_ = models[i].predict_(X_te)

		# 6. Calculate and display the fraction of correct predictions over the total number of predictions based on the test set and compare it to the train set.
		y_pred = np.argmax(y_pred_, axis=1).reshape(-1,1) + 1
		print("fraction of correct predictions for test data:  ", MyLR.score_(y_pred, y_test))

		
		# 7. Plot 3 scatter plots (one for each pair of citizen features) with the dataset and the final prediction of the model.
		_, fig = plt.subplots(1, 3, figsize=(24, 10))
		scatter_plot(fig[0], x_test[:, 0], x_test[:, 1], y_test.reshape(-1,), y_pred.reshape(-1,), labels[0], labels[1])
		scatter_plot(fig[1], x_test[:, 0], x_test[:, 2], y_test.reshape(-1,), y_pred.reshape(-1,), labels[0], labels[2])
		scatter_plot(fig[2], x_test[:, 2], x_test[:, 1], y_test.reshape(-1,), y_pred.reshape(-1,), labels[2], labels[1])
		plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n" \
			+ "Percentage of correct predictions for test data:  " +  str(round(100 * MyLR.score_(y_pred, y_test), 1)) + "%")
		plt.show()



	except Exception as e:
		print(e)