''' Author: Songheng Yin,
    Contact: songheng.yin@mail.utoronto.ca'''

import numpy as np
from matplotlib import pyplot as plt

def shuffle_data(data):
    ''' perform uniformly random permutation'''
    t, X = data
    n = t.shape[0]
    for i in range(n - 1):
        j = np.random.randint(i, n - 1)
        X[[i, j], :] = X[[j, i], :]
        t[[i, j]] = t[[j, i]]
    return (t, X)

def split_data(data, num_folds, fold):
    t, X = data
    X_split = np.array_split(X, num_folds, axis=0)
    t_split = np.array_split(t, num_folds)
    data_fold = (t_split[fold - 1], X_split[fold - 1])
    X_rest = np.delete(X_split, fold - 1, axis=0)
    X_rest = np.concatenate(X_rest, axis=0)
    t_rest = np.delete(t_split, fold - 1, axis=0)
    t_rest = np.concatenate(t_rest, axis=0)
    data_rest = (t_rest, X_rest)
    return data_fold, data_rest

def train_model(data, lambd):
    t, X = data
    N, D = X.shape
    w = np.linalg.inv((X.T @ X) + lambd * N * np.identity(D))
    w = (w @ X.T) @ t
    return w

def predict(data, model):
    t, X = data
    y = X @ model
    return y

def loss(data, model):
    t, X = data
    return (np.linalg.norm(X @ model - t)) ** 2 / (2 * t.shape[0])

def cross_validation(data, num_folds, lambd_seq):
    data = shuffle_data(data)
    cv_error = np.zeros(len(lambd_seq))
    for i, lambd in enumerate(lambd_seq):
        for fold in range(1, num_folds + 1):
            val_cv, train_cv = split_data(data, num_folds=num_folds, fold=fold)
            model = train_model(train_cv, lambd)
            cv_error[i] += loss(val_cv, model) / num_folds
    return cv_error

if __name__ == '__main__':
    data_train = (
        np.genfromtxt('./data/data_train_y.csv', delimiter=','),
        np.genfromtxt('./data/data_train_X.csv', delimiter=','),
    )
    data_test = (
        np.genfromtxt('./data/data_test_y.csv', delimiter=','),
        np.genfromtxt('./data/data_test_X.csv', delimiter=','),
    )
    lambd_seq = np.linspace(0.00005, 0.005, num=50)

    # Question c
    training_errors, test_errors = [], []
    for lambd in lambd_seq:
        model = train_model(data_train, lambd)
        test_errors.append(loss(data_test, model))
        training_errors.append(loss(data_train, model))
    fig, ax = plt.subplots(1, 1)
    ax.plot(lambd_seq, training_errors, label='Training Error')
    ax.plot(lambd_seq, test_errors, label='Test Error')

    # Question d
    cv_5_error = cross_validation(data_train, 5, lambd_seq)
    cv_10_error = cross_validation(data_train, 10, lambd_seq)
    ax.plot(lambd_seq, cv_5_error, label='CV 5 folders error')
    ax.plot(lambd_seq, cv_10_error, label='CV 10 folders error')
    ax.set_xlabel('Lambda')    
    ax.set_ylabel('Loss')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    best_lambda_5 = lambd_seq[np.argmin(cv_5_error)]
    best_lambda_10 = lambd_seq[np.argmin(cv_10_error)]
    print('Best lambda by 5-folder CV: {}'.format(best_lambda_5))
    print('Best lambda by 10-folder CV: {}'.format(best_lambda_10))