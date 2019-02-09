import random
import numpy as np
from scipy import stats

import sys
sys.path.insert(0, "../NumPy-based-0hl-Neural-Net/")
from numpy_based_0hl_neural_network import NumPyBased0hlNeuralNetwork

class EnsembleClassificationWrapper(object):

    def __init__(self, type="logistic_regression", configuration=None, number_models=10, debug_mode=False):
        self.__models = [None for i in range(number_models)]
        for i in range(number_models):
            if type == "logistic_regression":
                self.__models[i] = NumPyBased0hlNeuralNetwork()
            else:
                if debug_mode:
                    print("Error: unsupported type of model")
                    print("\tStack trace: EnsembleClassificationWrapper.__init__()")
                break


    def bag(self, X, Y, k, debug_mode=False):
        if X.shape[1] != Y.shape[1]:
            if debug_mode:
                print("Error: inconsistent number of examples")
                print("\tStack trace: EnsembleClassificationWrapper.bag()")
            return None
        # get the number of examples
        m = Y.shape[1]
        # sanity check: compare k and m
        if k > m:
            k = m
            if debug_mode:
                print("Warning: the number of bags, k, cannot be larger than the number of examples, m; k is automatically reset to m")
                print("\tStack trace: EnsembleClassificationWrapper.bag()")
        # bag indices of examples randomly without replacement into different bags
        index_pool = [i for i in range(m)]
        index_bags = []
        for i in range(k - 1):
            index_bag = []
            for j in range(m // k):
                index_chosen = random.choice(index_pool)
                index_pool.remove(index_chosen)
                index_bag.append(index_chosen)
            index_bags.append(index_bag)
        index_bags.append(index_pool)
        # bag examples into different bags based on index_bags
        X_bags = []
        Y_bags = []
        for index_bag in index_bags:
            X_bags.append(X[:, index_bag])
            Y_bags.append(Y[:, index_bag])
        return (X_bags, Y_bags)

    def fit(self, X, Y, batch_size=1, debug_mode=False):
        if X.shape[1] != Y.shape[1]:
            if debug_mode:
                print("Error: inconsistent number of examples")
                print("\tStack trace: EnsembleClassificationWrapper.fit()")
            return False
        X_bags, Y_bags = self.bag(X=X, Y=Y, k=len(self.__models), debug_mode=debug_mode)
        for i in range(len(self.__models)):
            X_bag = X_bags[i % len(X_bags)]
            Y_bag = Y_bags[i % len(Y_bags)]
            self.__models[i].fit(X=X_bag, Y=Y_bag, batch_size=batch_size, debug_mode=debug_mode)
        return True

    def predict(self, X, debug_mode=False):
        # initialize the NumPy array for ensemble prediction
        Y_ensemble = np.empty((len(self.__models), X.shape[1]))
        # each internal model make a prediction in regular representation
        for i in range(len(self.__models)):
            Y_regular = self.__models[i].predict(X=X, debug_mode=debug_mode)
            Y_ensemble[i, :] = Y_regular
        # use majority rule to make the final prediction
        Y_majority = np.array(stats.mode(Y_ensemble, axis=0)[0]).reshape(1, X.shape[1])
        return Y_majority
