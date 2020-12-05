from utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    # Given the weights and bias, compute the probabilities predicted   #
    # by the logistic classifier.                                       #
    #####################################################################
    # add the dummy 1's column to the data matrix
    one_column = np.ones(data.shape[0])
    data = np.column_stack((one_column, data))
    y = sigmoid(data @ weights)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y.reshape(-1, 1)


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    # Given targets and probabilities predicted by the classifier,      #
    # return cross entropy and the fraction of inputs classified        #
    # correctly.                                                        #
    #####################################################################
    # add a very small number to avoid log(0)
    ce = -np.mean((targets * np.log(y + 1e-7) + (1 - targets) * np.log(1 - y + 1e-7)))
    y_ = np.array(y, copy=True)
    for i in range(y_.shape[0]):
        y_[i, 0] = 0 if y_[i, 0] < 0.5 else 1
    frac_correct = 1 - (np.sum(np.abs(y_ - targets)) / y_.shape[0])
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points, gradient of parameters, and the probabilities given by    #
    # logistic regression.                                              #
    #####################################################################
    one_column = np.ones(data.shape[0])
    data = np.column_stack((one_column, data))
    f, _ = evaluate(targets, y)
    targets = targets.reshape(-1, 1)
    df = (data.T @ (y - targets)) / data.shape[0]
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (M+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)

    #####################################################################
    # TODO:                                                             #
    # Given weights and data, return the averaged loss over all data    #
    # points (plus a penalty term), gradient of parameters, and the     #
    # probabilities given by penalized logistic regression.             #
    #####################################################################
    one_column = np.ones(data.shape[0])
    data = np.column_stack((one_column, data))
    ce, _ = evaluate(targets, y)
    weights_ = np.array(weights, copy=True)
    weights_[-1] = 0  # NOT penalize the bias
    f = ce + (hyperparameters["weight_regularization"] / 2) * (np.linalg.norm(weights_) ** 2)
    df = (data.T @ (y - targets)) / data.shape[0] + hyperparameters["weight_regularization"] * weights_
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y
