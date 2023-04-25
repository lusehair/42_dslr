
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
class MyLogisticRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=1e-6, max_iter=10000, lambda_=0.5):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.array(thetas)
        self.lambda_ = lambda_



    def sigmoid_(self,x):
        """
        Compute the sigmoid of a vector.
        Args:
        x: has to be a numpy.ndarray of shape (m, 1).
        Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
        Raises:
        This function should not raise any Exception. 
        """ 
        return 1/(1 + np.exp(-x))
    
    
    def add_intercept(self, x):
        """Adds a column of 1’s to the non-empty numpy.array x.
        Args:
        x: has to be a numpy.array of dimension m * n.
        Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
        Raises:
        This function should not raise any Exception.
        """ 
        if not isinstance(x, np.ndarray) or x.size == 0:
            return None
        b = np.ones(len(x))
        return np.c_[b,x]
    
    def simple_gradient(self, x, y, theta):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """

        if not isinstance(x, np.ndarray) or x.size == 0 : 
            return None 
        if not isinstance(y, np.ndarray) or y.size == 0 : 
            return None 
        if not isinstance(theta, np.ndarray) or theta.size == 0 : 
            return None 
        if x.shape != y.shape or theta.shape != (2, 1) : 
            return None 
        if x.shape != (x.shape[0], 1) or y.shape != (y.shape[0], 1) : 
            return None 


        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1) 
        theta = theta.reshape(-1, 1) 
        x = self.add_intercept(x) 
        m = len(y) 
        return x.T.dot(x.dot(theta) -y ) / m




    def predict_(self, x):
       
        x = self.add_intercept(x)
        return self.sigmoid_(np.dot(x, self.theta))


    def fit_(self, x, y):
        
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exception.
        """
        theta = self.theta
        if not isinstance(x, np.ndarray) or x.size == 0 : 
            return None 
        if not isinstance(y, np.ndarray) or y.size == 0 : 
            return None 
        if not isinstance(self.theta, np.ndarray) or theta.size == 0 : 
            return None 
        if not isinstance(self.alpha, float) : 
            return None 
        if not isinstance(self.max_iter, int) : 
            return None 
        if x.shape[0] != y.shape[0] : 
            return None 
        if x.shape[1] + 1 != theta.shape[0] : 
            return None 
       

        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        theta_prime = np.array(self.theta, copy=True)
        theta_prime[0] = 0
        for _ in range(self.max_iter):
            gradient = (
                x.T @ (self.sigmoid_(x @ self.theta) - y)
                + self.lambda_ * theta_prime
            ) / x.shape[0]
            self.theta -= self.alpha * gradient
        return self.theta
        




    def loss_elem_(self,y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """ 
        if not isinstance(y, np.ndarray) or y.size == 0 : 
            return None 
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
            return None 
       
        
        y_true = y 
        y_pred = y_hat 
        y_zero_loss = y_true * np.log(y_pred + eps)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + eps)
        return -np.mean(y_zero_loss + y_one_loss) 


    def loss_(self, y, y_hat):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """ 
        if not isinstance(y, np.ndarray) or y.size == 0 : 
            return None 
        if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
            return None 

        eps = 1e-15
        ones = np.ones(y.shape)
        res =  np.sum(y * np.log(y_hat + eps) + (ones - y) * np.log(ones - y_hat + eps)) / -y.shape[0]
        return res

def data_spliter(x, y, proportion):
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
    len_xtest = int((len(x) / proportion) - len(x))
    len_ytest = int((len(y) /proportion) -len(y)) 
  
    np.random.shuffle(x) 
    np.random.shuffle(y) 

    x_train = x[:len(x)-len_xtest] 
    y_train = y[:len(y)-len_ytest]
    x_test = x[len(x)-len_xtest:] 
    y_test = y[len(y)-len_ytest:]
    return x_train, x_test, y_train, y_test 

def zscore(x):
    if not isinstance(x, np.ndarray) or x.size == 0 : 
        return None 
    if x.shape != (x.shape[0], 1) :
        x = x.reshape(-1, 1)
    ret = np.zeros(x.shape)
    for i, el in np.ndenumerate(x) : 
        ret[i] = (el - np.mean(x)) /  np.std(x)
    return  ret


def accuracy_score_(y, y_hat):
    return np.sum(y == y_hat) / y.shape[0]
    
def precision_score_(y, y_hat, pos_label=1):
    if not isinstance(y, np.ndarray) or y.size == 0 : 
        return None 
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
        return None 
    if not isinstance(pos_label, (int, str)) : 
        return None 
    if y.shape != y_hat.shape : 
        return None 
    if pos_label not in y : 
        return None 
    if pos_label not in y_hat : 
        return None 
    
    y = y.reshape(-1, 1) 
    y_hat = y_hat.reshape(-1, 1) 
    TP = np.sum((y == pos_label) & (y_hat == pos_label)) 
    FP = np.sum((y != pos_label) & (y_hat == pos_label)) 
    return TP / (TP + FP) 


def recall_score_(y, y_hat, pos_label=1):

    if not isinstance(y, np.ndarray) or y.size == 0 : 
        return None 
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
        return None 
    if not isinstance(pos_label, (int, str)) : 
        return None 
    if y.shape != y_hat.shape : 
        return None 
    if pos_label not in y : 
        return None 
    if pos_label not in y_hat : 
        return None 
    
    y = y.reshape(-1, 1) 
    y_hat = y_hat.reshape(-1, 1) 
    TP = np.sum((y == pos_label) & (y_hat == pos_label)) 
    FN = np.sum((y == pos_label) & (y_hat != pos_label)) 
    return TP / (TP + FN)

def f1_score_(y, y_hat, pos_label=1):

    if not isinstance(y, np.ndarray) or y.size == 0 : 
        return None 
    if not isinstance(y_hat, np.ndarray) or y_hat.size == 0 : 
        return None 
    if not isinstance(pos_label, (int, str)) : 
        return None 
    if y.shape != y_hat.shape : 
        return None 
    if pos_label not in y : 
        return None 
    if pos_label not in y_hat : 
        return None 
    
    y = y.reshape(-1, 1) 
    y_hat = y_hat.reshape(-1, 1) 
    TP = np.sum((y == pos_label) & (y_hat == pos_label)) 
    FP = np.sum((y != pos_label) & (y_hat == pos_label)) 
    FN = np.sum((y == pos_label) & (y_hat != pos_label)) 
    precision = TP / (TP + FP) 
    recall = TP / (TP + FN) 
    return 2 * precision * recall / (precision + recall) 



if __name__ == "__main__" : 
    df = pd.read_csv( "dataset_test.csv") 
    df = df.dropna(axis=0, how='any')
    # in df keep only "Charms", "Ancient Runes", "Herbology", "Astronomy", "Hogwarts House", "index" 
    df = df.drop( [ "Arithmancy" , "Defense Against the Dark Arts" , "Divination" , "Muggle Studies" , "History of Magic" , "Transfiguration" , "Potions" , "Care of Magical Creatures" , "Flying" ] , axis = 1 ) 
    df = df.drop( ["First Name" , "Last Name" , "Best Hand", "Birthday"] , axis = 1) 
    df = df.drop(["Hogwarts House", "index"], axis = 1) 

    # Y is the target and the Hogwart House 
    # Is like the range(4) 
    #y = np.array('Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff')
    Houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'] 
    y = Houses 
    # y = y.reshape(-1, 1) 
    # x is the features is Charms, Ancient Runes, Herbology, Astronomy  
    x = np.array(df[['Charms', 'Ancient Runes', 'Herbology', 'Astronomy']]) 

    nb = 0 
    f1_score = [] 
    f1_val = [] 
    models = np.genfromtxt("theta.csv", dtype=str, delimiter="\t")
    theta = []
    for el in models : 
        theta.append(np.fromstring(el, sep=","))
    theta = theta[1:]

    for j in range(6) : 
        for zipcode in range(4) : 
            my_lreg = MyLogisticRegression(theta[zipcode + nb].reshape(-1, 1), lambda_=j / 5)  
        x_pred = np.insert(x, 0, values=1.0, axis=1) 
        y_hat = np.array([ max((np.dot(i, np.array(theta[zipcode + nb])), zipcode) for zipcode in range(4))[1] for i in x_pred]).reshape(-1, 1) 
        
        x_valpred = np.insert(x, 0, values=1.0, axis=1).astype(float) 
        y_hatval = np.array([ max(( np.dot(i, np.array(theta[zipcode + nb])), zipcode) for zipcode in range(4))[1] for i in x_valpred]).reshape(-1, 1)
        f1_val.append(f1_score_(y, y_hatval))
        print("the val is : ", f1_val[j]) 
        nb+= 4 
  

    _lambda = f1_val.index(max(f1_val)) / 5 
    print("the best lambda is : ", _lambda) 
    theta = [] 
    for zipcode in range(4) : 
        my_lreg = MyLogisticRegression(np.ones(x.shape[1] + 1).reshape(-1, 1), lambda_=_lambda) 
        y_one_loss = np.where(y == zipcode, 1, 0)
        my_lreg.fit_(x, y_one_loss)   
        theta.append(my_lreg.theta) 
    x_pred = np.insert(x, 0, values=1.0, axis=1).astype(float) 
    y_hat = np.array([ max((np.dot(i, np.array(theta[zipcode])), zipcode) for zipcode in range(4))[1] for i in x_pred]).reshape(-1, 1)
    x_predtest = np.insert(x, 0, values=1.0, axis=1).astype(float) 
    y_hattest = np.array([ max((np.dot(i, np.array(theta[zipcode])), zipcode) for zipcode in range(4))[1] for i in x_predtest]).reshape(-1, 1) 
    
   




