import numpy as np
from ensemble_classification_wrapper import EnsembleClassificationWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore")

"""
Read data from file

@param filename: (optional) the name of the file we want to read the data from; the default value if "good-moves.txt"
@return data: the NumPy array of the attributes, shape = (number of examples, number of attributes)
@return target: the NumPy array of the labels, shape = (number of examples, 1)
"""
def read_from_file(filename="good-moves.txt"):
    examples = []
    with open(filename, "r") as file:
        for line in file:
            example = []
            for letter in line.strip():
                digit = int(letter)
                example.append(digit)
            example = np.array(example)
            examples.append(example)
    examples = np.array(examples)
    data = examples[:, 0: -1]
    target = examples[:, -1: examples.shape[1]]
    return (data, target)

"""
Convert the NumPy array of target from regular to one-hot representation

@param target_regular: the NumPy array of target in regular representation, shape = (number of examples, 1)
@return target_onehots: the NumPy array of target in one-hot representation, shape = (number of examples, number of classes)
"""
def convert_target_from_regular_to_onehots(target_regular):
    # find the minimum and maximum class labels
    min_class = np.min(target_regular)
    max_class = np.max(target_regular)
    # initialize a NumPy array full of zeros, shape = (number of examples, number of classes)
    target_onehots =np.zeros((target_regular.shape[0], max_class - min_class + 1))
    # populate the NumPy array with ones at the correct cell
    for row_index in range(target_regular.shape[0]):
        col_index = target_regular[row_index, 0] - min_class
        target_onehots[row_index, col_index] = 1
    return target_onehots

"""
Convert the NumPy array of target from one-hot to regular representation

@param target_onehots: the NumPy array of target in one-hot representation, shape = (number of examples, number of classes)
@return target_regular: the NumPy array of target in regular representation, shape = (number of examples, 1)
"""
def convert_target_from_onehots_to_regular(target_onehots):
    target_regular = np.argmax(target_onehots, axis=0).reshape((1, target_onehots.shape[1]))
    return target_regular

def main(debug_mode=True):
    X, Y_regular = read_from_file()
    Y_onehots = convert_target_from_regular_to_onehots(Y_regular)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehots, test_size=0.2, random_state=0)
    X_train, X_test, Y_train, Y_test = X_train.T, X_test.T, Y_train.T, Y_test.T
    if debug_mode:
        print("X_train.shape:\t " + str(X_train.shape))
        print("X_test.shape:\t " + str(X_test.shape))
        print("Y_train.shape:\t " + str(Y_train.shape))
        print("Y_test.shape:\t " + str(Y_test.shape))
    ensemble_classification_wrapper = EnsembleClassificationWrapper(type="logistic_regression", configuration=None, number_models=10, debug_mode=debug_mode)
    ensemble_classification_wrapper.fit(X=X_train, Y=Y_train, batch_size=100, debug_mode=debug_mode)
    Y_majority = ensemble_classification_wrapper.predict(X_test)
    Y_test_regular = convert_target_from_onehots_to_regular(Y_test)
    print(Y_majority.shape)
    print(Y_test_regular.shape)
    F1_score = f1_score(Y_test_regular.T, Y_majority.T, average="weighted")
    print("Test set F1 score = " + str(F1_score))

main()
