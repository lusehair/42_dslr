import numpy as np
import matplotlib.pyplot as plt
from random import randint


def data_splitter(x, y, proportion):
	"""Shuffles and splits the dataset (given by x and y) into a training and a test set,
	while respecting the given proportion of examples to be kept in the training set.
	Args:
	x: has to be an numpy.array, a matrix of dimension m * n.
	y: has to be an numpy.array, a vector of dimension m * 1.
	proportion: has to be a float, the proportion of the dataset that will be assigned to the
	training set.
	Return:
	(x_train, x_test, y_train, y_test) as a tuple of numpy.array
	None if x or y is an empty numpy.array.
	None if x and y do not share compatible dimensions.
	None if x, y or proportion is not of expected type.
	Raises:
	This function should not raise any Exception.
	"""
	try:
		assert isinstance(
			x, np.ndarray), "1st argument must be a numpy.ndarray, a vector of dimension m * n"
		assert isinstance(
			y, np.ndarray) and (y.ndim == 1 or y.ndim == 2),  "2nd argument must be a numpy.ndarray, a vector of dimension m * 1"
		m = x.shape[0]
		n = x.shape[1]
		if (x.ndim == 1):
			x = x.reshape(-1, 1)
		if (y.ndim == 1):
			y = y.reshape(-1, 1)
		assert y.shape[0] == x.shape[0], "arrays must be the same size"
		assert np.any(x) or np.any(y), "arguments cannot be empty numpy.ndarray"
		assert isinstance(
			proportion, float) and proportion > 0 and proportion < 1, "3rd argument max_iter must be a positive float between 0 and 1"

		# shuffle input data
		x_shuffle = x[:]
		y_shuffle = y[:]
		rdi = randint(0, 100)
		np.random.seed(rdi)
		np.random.shuffle(x_shuffle, )
		np.random.seed(rdi)
		np.random.shuffle(y_shuffle, )

		# extract splitted data
		idx = int(m * proportion)
		x_train = x_shuffle[:idx]
		x_test = x_shuffle[idx:]
		y_train = y_shuffle[:idx]
		y_test = y_shuffle[idx:]

		return (x_train, x_test, y_train, y_test)

	except Exception as e:
		print(e)
		return None