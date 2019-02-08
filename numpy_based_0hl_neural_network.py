import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

"""
A 0-hidden-layer neural network classifier that optimizes the parameters using
gradient descent, uses the sigmoid activation function to compute the probability
of each class, and predict a label based on maximum a posterior probability
"""
class NumPyBased0hlNeuralNetwork(object):

    """
    Constructor: declare member variables

    @return a new instance of this object
    """
    def __init__(self):
        # the number of features
        self.__n_x = None
        # the number of labels
        self.__n_y = None
        # the NumPy array of the weights
        self.__W = None
        # the NumPy array of the bias terms
        self.__b = None
        # the log of epoch cost
        self.__epoch_costs = None
        # the log of iterative cost
        self.__iterative_costs = None

    """
    Get the cross-entropy cost of the model based on the provided data

    @param Y: the NumPy array of the actual labels in one-hot representation, shape = (n_y, m)
    @param A: the NumPy array of the log probabilitie of each label, shape = (n_y, m)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return the cross-entropy cost of the model based on the provided data
    """
    def __get_cost(self, Y, A, debug_mode=True):
        # check the dimension of Y & A
        if Y.shape != A.shape:
            if debug_mode:
                print("Error: Y.shape != A.shape")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__get_cost()")
            return None
        # tge the numbe of examples m
        m = Y.shape[1]
        # calcualte cross-entropy cost
        loss = - np.multiply(Y, np.log(A)) - np.multiply((1.0 - Y), (1.0 - A))
        cost = np.squeeze((1.0 / m) * np.sum(loss))
        return cost

    """
    The sigmoid function

    @param Z: the NumPy array of original values, shape = (n_y, m)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return the NumPy array of sigmoid values
    """
    def __sigmoid(self, Z, debug_mode=True):
        A = 1.0 / (1.0 + np.exp(Z))
        return A

    """
    The derivative of the sigmoid function in one-hot representation

    @param A: the NumPy array of the activated values, shape = (n_y, m)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return the NumPy array of derivatives
    """
    def __sigmoid_derivative(self, A, debug_mode=True):
        derivative = np.multiply(A, (1.0 - A))
        return derivative

    """
    The forward propagation module

    @param W: the NumPy array of the weights, shape = (n_y, n_x)
    @param X: the NumPy array of the inputs, shape = (n_x, m)
    @param b: the NumPy array of the bias terms, shape = (n_y, 1)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a tuple which contains (1) a NumPy array of the linear outputs, and (2) a NumPy array of the activated values
    """
    def __forward_propagation(self, W, X, b, debug_mode=True):
        # check the number of features
        if W.shape[1] != X.shape[0]:
            if debug_mode:
                print("Error: inconsistent number of features")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__forward_propagation()")
            return None
        # check the number of labels
        if W.shape[0] != b.shape[0]:
            if debug_mode:
                print("Error: inconsistent number of labels")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__forward_propagation()")
            return None
        # calcualte the inear output
        Z = np.dot(W, X) + b
        # calcualte the activated term
        A = self.__sigmoid(Z=Z, debug_mode=debug_mode)
        return (Z, A)

    """
    Get the gradients of the model based on the provided data

    @param X: the NumPy array of the inputs, shape = (n_x, m)
    @param Y: the NumPy array of the actual labels in one-hot representation, shape = (n_y, m)
    @param Z: the NumPy array of the linear outputs, shape = (n_y, m)
    @param A: the NumPy array of the activated values, shape = (n_y, m)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a tuple which contains (1) a NumPy array which contains the gradient of the activated terms, (2) a NumPy array which contains the gradient of the linear outputs, (3) a NumPy array which contains the gradient of the weights, and (4) the gradient of the bias terms
    """
    def __get_gradients(self, X, Y, Z, A, debug_mode=True):
        # check the number of examples
        if X.shape[1] != Y.shape[1] or Y.shape[1] != Z.shape[1] or Z.shape[1] != A.shape[1]:
            if debug_mode:
                print("Error: inconsistent number of examples")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__get_gradients()")
            return None
        # check the number of labels
        if Y.shape[0] != Z.shape[0] or Z.shape[0] != A.shape[0]:
            if debug_mode:
                print("Error: inconsistent number of labels")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__get_gradients()")
            return None
        # get the number of examples m
        m = Y.shape[1]
        # calcualte the gradient of the activated term dA
        dA = - (np.divide(Y, A) - np.divide((1.0 - Y), (1.0 - A)))
        # calculate the gradient of the linear output dZ
        dZ = np.multiply(dA, self.__sigmoid_derivative(Z))
        # calcualte the gradient of the weights dW
        dW = (1.0 / m) * np.dot(dZ, X.T)
        # calcualte the gradient of the bias term db
        db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        return (dA, dZ, dW, db)

    """
    Update the parameters of the model

    @param W: the NumPy array of the weights, shape = (n_y, n_x)
    @param b: the NumPay array of the bias terms, shape = (n_y, 1)
    @param learning_rate: the speed of gradient descent
    @param dW: the NumPy array of the gradient of the weights, shape = (n_y, n_x)
    @param db: the NumPy array of the gradient of the bias terms, shape = (n_y, 1)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a tuple which contains (1) a NumPy array of the updated weights, and (2) a NumPy array of the updated bias terms
    """
    def __update_parameters(self, W, b, learning_rate, dW, db, debug_mode=True):
        # check the dimension of W & dW
        if W.shape != dW.shape:
            if debug_mode:
                print("Error: W.shape != dw.shape")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__update_parameters()")
            return None
        # check the dimension of b & db
        if b.shape != db.shape:
            if debug_mode:
                print("Error: b.shape != db.shape")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.__update_parameters()")
            return None
        # update the weights and the bias term using gradient descent
        W -= learning_rate * dW
        b -= learning_rate * db
        return (W, b)

    """
    Get the gradients of the model based on the provided data

    @param X: the NumPy array of the inputs, shape = (n_x, m)
    @param Y: the NumPy array of the actual labels in one-hot representation, shape = (n_y, m)
    @param Z: the NumPy array of the linear outputs, shape = (n_y, m)
    @param A: the NumPy array of the activated values, shape = (n_y, m)
    @param W: the NumPy array of the weights, shape = (n_y, n_x)
    @param b: the NumPay array of the bias terms, shape = (n_y, 1)
    @param learning_rate: the speed of gradient descent
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is true
    @return a tuple which contains (1) a NumPy array of the updated weights, and (2) a NumPy array of the updated bias terms
    """
    def __backward_propagation(self, X, Y, Z, A, W, b, learning_rate, debug_mode=True):
        # get the gradients
        dA, dZ, dW, db = self.__get_gradients(X=X, Y=Y, Z=Z, A=A, debug_mode=debug_mode)
        # if debug_mode:
        #     print("numpy_based_0hl_neural_network.__backward_propagation.dA.shape = " + str(dA.shape))
        #     print("numpy_based_0hl_neural_network.__backward_propagation.dZ.shape = " + str(dZ.shape))
        #     print("numpy_based_0hl_neural_network.__backward_propagation.dW.shape = " + str(dW.shape))
        #     print("numpy_based_0hl_neural_network.__backward_propagation.db.shape = " + str(db.shape))
        # update the parameters
        W, b = self.__update_parameters(W=W, b=b, learning_rate=learning_rate, dW=dW, db=db, debug_mode=debug_mode)
        return (W, b)

    """
    Fit the model to the provided data

    @param X: the NumPy array of the inputs, shape = (n_x, m)
    @param Y: the NumPy array of the actual labels in one-hot representation, shape = (n_y, m)
    @param learning_rate: (optional) the speed of gradient descent; the default value is 0.001
    @param early_stopping_point: (optional) the maximum number of epoches allowed; the default value is 1000
    @param convergence_tolerance: (optional) the threshold to decide whether the gradient descent converges; the default value is 0.001
    @param batch_size: (optional) the batch size of mini-batch gradient descent; the default value is 1
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
    @param loss_plot_mode: (optional) a boolean value that indicates whether the loss plot mode is active; the default value is false
    @return a boolean value that indicates whether the fitting is successful
    """
    def fit(self, X, Y, learning_rate=0.001, decay_rate=0.1, early_stopping_point=1000, convergence_tolerance=0.001, batch_size=1, debug_mode=False, cost_plot_mode=False):
        # check the number of examples
        if X.shape[1] != Y.shape[1]:
            if debug_mode:
                print("Error: inconsistent number of examples")
                print("Stack trace: NumPyBased0hlNeuralNetwork.fit()")
            return False
        # reconfigure the model setting
        self.__n_x = X.shape[0]
        self.__n_y = Y.shape[0]
        self.__W = np.random.randn(self.__n_y, self.__n_x)
        self.__b = np.random.randn(self.__n_y, 1)
        self.__epoch_costs = []
        self.__iterative_costs = []
        if debug_mode:
            print("NumPyBased0hlNeuralNetwork.__n_x = " + str(self.__n_x))
            print("NumPyBased0hlNeuralNetwork.__n_y = " + str(self.__n_y))
            print("NumPyBased0hlNeuralNetwork.__W.shape = " + str(self.__W.shape))
            print("NumPyBased0hlNeuralNetwork.__b.shape = " + str(self.__b.shape))
        # allocate batches
        m = Y.shape[1]
        if batch_size > m:
            batch_size = m
        elif batch_size < 1:
            batch_size = 1
        num_batches = m // batch_size + (m % batch_size > 0)
        X_batches = np.array_split(X, num_batches, axis=1)
        Y_batches = np.array_split(Y, num_batches, axis=1)
        # epoches of gradient descent
        for epoch in range(early_stopping_point):
            # get the epoch cost and add to epoch fitting log
            Z, A = self.__forward_propagation(W=self.__W, X=X, b=self.__b, debug_mode=debug_mode)
            epoch_cost = self.__get_cost(Y=Y, A=A, debug_mode=debug_mode)
            if debug_mode:
                print("Epoch {:8s}".format(str(epoch) + ":") + "cost = " + str(epoch_cost))
            self.__epoch_costs.append(epoch_cost)
            if epoch >= 2 and abs(self.__epoch_costs[-1] - self.__epoch_costs[-2]) < convergence_tolerance:
                if debug_mode:
                    print("Message: convergence_tolerance reached at epoch " + str(epoch))
                    print("\tStack trace: NumPyBased0hlNeuralNetwork.fit()")
                break
            # iterate through batches
            for batch_index in range(num_batches):
                # get the batch based on batch index
                X_batch = X_batches[batch_index]
                Y_batch = Y_batches[batch_index]
                # get the iterative cost and add to the iterative fitting log
                Z_batch, A_batch = self.__forward_propagation(W=self.__W, X=X_batch, b=self.__b, debug_mode=debug_mode)
                iterative_cost = self.__get_cost(Y=Y_batch, A=A_batch, debug_mode=debug_mode)
                self.__iterative_costs.append(iterative_cost)
                # backward propagation
                self.__W, self.__b = self.__backward_propagation(X=X_batch, Y=Y_batch, Z=Z_batch, A=A_batch, W=self.__W, b=self.__b, learning_rate=learning_rate, debug_mode=debug_mode)
        if cost_plot_mode:
            # plot epoch costs
            plt.plot(self.__epoch_costs)
            plt.title("NumPy-based 0-hidden-layer Neural Network, batch size = " + str(batch_size) + "\nEpoch cross-entropy costs\nErnest Xu")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy cost")
            plt.show()
            # plot epoch costs
            plt.plot(self.__iterative_costs)
            plt.title("NumPy-based 0-hidden-layer Neural Network, batch size = " + str(batch_size) + "\nIterative cross-entropy costs\nErnest Xu")
            plt.xlabel("Iteration")
            plt.ylabel("Cross-entropy cost")
            plt.show()
        return True

    """
    Fit the model to the provided data

    @param X: the NumPy array of the inputs, shape = (n_x, m)
    @param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
    @return a NumPy array of the predicted labels in regular representation
    """
    def predict(self, X, debug_mode=False):
        # check number of features
        if X.shape[0] != self.__n_x:
            if debug_mode:
                print("Error: inconsistent number of features")
                print("\tStack trace: NumPyBased0hlNeuralNetwork.predict()")
            return None
        # forward propagation
        Z, A = self.__forward_propagation(W=self.__W, X=X, b=self.__b, debug_mode=debug_mode)
        # make classification
        predicted_classes = np.argmax(A, axis=0)
        return predicted_classes
