from l2_distance import l2_distance
from utils import *

import matplotlib.pyplot as plt
import numpy as np


def knn(k, train_data, train_labels, valid_data):
    """ Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          M is the number of features per example.

    :param k: The number of neighbours to use for classification
    of a validation example.
    :param train_data: N_TRAIN x M array of training data.
    :param train_labels: N_TRAIN x 1 vector of training labels
    corresponding to the examples in train_data (must be binary).
    :param valid_data: N_VALID x M array of data to
    predict classes for validation data.
    :return: N_VALID x 1 vector of predicted labels for
    the validation data.
    """
    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:, :k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # Note this only works for binary labels:
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1, 1)

    return valid_labels


def run_knn():
    train_inputs, train_targets = load_train()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    x_axis, valid_axis, test_axis = [1, 3, 5, 7, 9], [], []
    for k in x_axis:
        # compute the validation performance
        valid_pred = knn(k, train_inputs, train_targets, valid_inputs)
        accuracy = 1 - (np.sum(np.abs(valid_pred - valid_targets)) / valid_targets.shape[0])
        valid_axis.append(accuracy)
        # Similarly, compute the test performance
        test_pred = knn(k, train_inputs, train_targets, test_inputs)
        accuracy = 1 - (np.sum(np.abs(test_pred - test_targets)) / test_targets.shape[0])
        test_axis.append(accuracy)
    
    # plot the graph
    print(valid_axis)
    fig, ax = plt.subplots(1, 1)
    ax.bar(x_axis, valid_axis)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Classification Rate')
    plt.xticks(x_axis, x_axis)
    plt.title('2.1(a)')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.bar(x_axis, test_axis)
    ax.set_xlabel('K Value')
    ax.set_ylabel('Classification Rate')
    plt.xticks(x_axis, x_axis)
    plt.title('2.1(b)')
    plt.show()
    plt.close()


if __name__ == "__main__":
    run_knn()
