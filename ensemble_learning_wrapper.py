import random
import numpy as np
from scipy import stats
from numpy_based_0hl_neural_network import NumPyBased0hlNeuralNetwork

class EnsembleClassificationWrapper(object):

    def __init__(self, type="NumPyBased0hlNeuralNetwork", number_models=10, debug_mode=False):
        self.__models = [None for i in range(number_models)]
        for i in range(number_models):
            if type == "NumPyBased0hlNeuralNetwork":
                self.__models.append(NumPyBased0hlNeuralNetwork())


    def bag(self, m, k=10):
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
        return index_bags

    def fit(self, X, Y, batch_size=1, debug_mode=False):
        merged_dataset = np.concatenate(X, Y, axis=0)
        m = merged_dataset.shape[1]
        bagged_dataset = []
        index_bags = self.bag(m=m, k=len(self.__models))
        for index_bag in index_bags:
            bagged_dataset.append(merged_dataset[:, index_bag])
        for i in range(len(self.__models)):
            X_bagged = bagged_dataset[i][0:X.shape[0], m]
            Y_bagged = bagged_dataset[i][X.shape[0]: X.shape[0] + Y.shape[0], m]
            self.__models.fit(X=X_bagged, Y=Y_bagged, batch_size=batch_size, debug_mode=debug_mode)

    def predict(self, X, debug_mode=False):
        Y_ensemble = np.empty((len(self.__models), X.shape[1]))
        for i in range(len(self.__models)):
            Y_regular = model.predict(X=X, debug_mode=debug_mode)
            Y_ensemble[i, :] = Y_regular
        Y_majority = stats.mode(Y_ensemble, axis=0)
        return Y_majority

wrapper = EnsembleClassificationWrapper()
print(wrapper.bag(m=145, k=10))
