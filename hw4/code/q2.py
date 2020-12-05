'''
Question 2 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for k in range(10):
        X = data.get_digits_by_label(train_data, train_labels, k)
        means[k] = np.sum(X, axis=0) / X.shape[0]
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class
    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    means = compute_mean_mles(train_data, train_labels)
    for k in range(10):
        X = data.get_digits_by_label(train_data, train_labels, k)
        covariances[k] = ((X - means[k]).T @ (X - means[k])) / X.shape[0] + 0.01 * np.identity(64)
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    res = np.zeros((digits.shape[0], 10))
    for k in range(10):
        temp = ((2 * np.pi) ** (-digits.shape[1] / 2)) * (np.linalg.det(covariances[k]) ** (-1/2)) * \
            np.exp(-0.5 * np.diag((digits - means[k]) @ np.linalg.inv(covariances[k]) @ (digits - means[k]).T))
        res[:, k] = np.log(temp)
    return res

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    numerator = generative_likelihood(digits, means, covariances) + np.log(0.1)
    denominator = logsumexp(numerator, axis=1).reshape(-1, 1)
    return numerator - denominator

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    counter = 0
    for i in range(digits.shape[0]):
        counter += cond_likelihood[i][int(labels[i])]
    # Compute as described above and return
    return counter / digits.shape[0]

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    # Question 1, report avg cond loglikelihood
    print('avg cond loglikelihood on training set = {}.'.format(avg_conditional_likelihood(train_data, train_labels, means, covariances)))
    print('avg cond loglikelihood on test set = {}.'.format(avg_conditional_likelihood(test_data, test_labels, means, covariances)))
    
    # Question 2, predict
    train_pred = classify_data(train_data, means, covariances)
    train_accuracy = np.sum(np.equal(train_pred, train_labels)) / train_labels.shape[0]
    print('Training accuracy = {}.'.format(train_accuracy))
    test_pred = classify_data(test_data, means, covariances)
    test_accuracy = np.sum(np.equal(test_pred, test_labels)) / test_labels.shape[0]
    print('Test accuracy = {}.'.format(test_accuracy))

    # Question 3, Compute eigenvectors
    for k in range(10):
        value, vector = np.linalg.eig(covariances[k])
        # print('leading eigenvectors for class {}: {}.'.format(k, vector[np.argmax(value)]))
        plt.imshow(vector[:, np.argmax(value)].reshape(8, 8))
        plt.savefig('./{}.png'.format(k))

if __name__ == '__main__':
    main()