from utils import *

import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    usr = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    para = theta[usr] - beta[q]
    log_like = np.log(sigmoid(para)) * c + np.log(1 - sigmoid(para)) * (1 - c)
    log_lklihood = np.sum(log_like)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    usr = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    para = theta[usr] - beta[q]
    values = c - sigmoid(para)

    sparse = csc_matrix((values, (usr, q)), shape=(len(theta), len(beta))).toarray()
    theta = theta + np.sum(sparse, axis=1) * lr

    para = theta[usr] - beta[q]
    values = c - sigmoid(para)
    sparse = csc_matrix((values, (usr, q)), shape=(len(theta), len(beta))).toarray()
    beta = beta - np.sum(sparse, axis=0) * lr
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(542)
    beta = np.zeros(1774)

    val_acc_lst = []
    train_loglik = []
    valid_loglik = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta)
        train_loglik.append(-neg_lld)
        valid_loglik.append(-val_neg_lld)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_loglik, valid_loglik


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, val_acc, train_loglik, valid_loglik \
        = irt(train_data, val_data, 0.01, 50)
    # plt.plot(val_acc)
    # plt.show()
    plt.plot(train_loglik, label="Training Log-likelihood")
    plt.plot(valid_loglik, label="Validation Log-likelihood")
    plt.xlabel("Num of Iteration")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)
    validation_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)
    print("Final validation accuracy is: " + str(validation_accuracy))
    print("Final test accuracy is: " + str(test_accuracy))
    #####################################################################
    # implement part (d)
    p_1 = sigmoid(theta - beta[0])
    p_2 = sigmoid(theta - beta[1])
    p_3 = sigmoid(theta - beta[2])
    p_4 = sigmoid(theta - beta[3])
    p_5 = sigmoid(theta - beta[4])

    plt.plot(theta, p_1, label="question 1")
    plt.plot(theta, p_2, label="question 2")
    plt.plot(theta, p_3, label="question 3")
    plt.plot(theta, p_4, label="question 4")
    plt.plot(theta, p_5, label="question 5")
    plt.xlabel("Value of theta")
    plt.ylabel("Probability of the correct response")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
