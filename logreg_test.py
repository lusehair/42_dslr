import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from my_logistic_regression import MyLogisticRegression as MyLR
from data_splitter import data_splitter
from scaler import Standard_Scaler
from logreg_train import num_houses, label_houses, fill_zeros, relabel, scatter_plot, mean_, median_
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

		# labels = ['Herbology', 'Defense Against the Dark Arts', 'Divination', 'Ancient Runes', 'Charms']
		labels = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms']
		# labels = ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Charms']
		# labels = ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Transfiguration', 'Charms']
		# labels = ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Charms', 'Flying']
		# labels = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms', 'Flying']
		# labels = ['Astronomy', 'Herbology', 'Divination', 'Ancient Runes', 'Transfiguration', 'Charms', 'Flying']
		# labels = ['Astronomy', 'Herbology', 'Ancient Runes', 'Charms', 'Flying']
		labels = ['Astronomy', 'Herbology', 'Defense Against the Dark Arts', 'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Charms', 'Flying']
		

		# Replace NaN value by mean
		mean_train = []
		for col in data_train[labels]:
			m = mean_(data_train[col])
			# m = median_(data_train[col])
			mean_train.append(m)
			data_train[col].fillna(m, inplace=True)
			data_testX[col].fillna(m, inplace=True)
		
		x = data_train[labels]
		x_train = x.values

		x = data_testX[labels]
		y = data_testY[['Hogwarts House']].values

		x_test = x.values
		
		features = labels
		houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		
		# 2. numerize y labels
		y_test = num_houses(y)

		# 3. Normalization
		# Zscore
		my_Scaler = Standard_Scaler()
		my_Scaler.fit(x_train)
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
		_, fig = plt.subplots(1, sum(range(len(labels))), figsize=(30, 10), constrained_layout = True)
		cnt = set()
		k = 0
		for i in range(len(labels)):
			for j in range(len(labels)):
				if i != j and ((i, j) not in cnt) and i < j:
					cnt.add((i, j))
					scatter_plot(fig[k], x_test[:, i], x_test[:, j], y_test.reshape(-1,), y_pred.reshape(-1,), labels[i], labels[j])
					k += 1
		fig[k - 1].legend(bbox_to_anchor=(1, 1), borderaxespad=1)
		plt.suptitle("Scatter plots with the dataset and the final prediction of the model\n" \
			+ "Percentage of correct predictions for test data:  " +   str(round(100 * MyLR.score_(y_pred, y_test), 1)) + "%\n" + "Labels:  " + str(labels))

		# 8. denumerize predictions
		houses_pred = label_houses(y_pred)
		df = pd.DataFrame(data=houses_pred, columns=['Hogwarts House'])
		df.index.name = 'Index'
		df.to_csv('houses.csv', index=True)
		plt.show()


	except Exception as e:
		print(e)
