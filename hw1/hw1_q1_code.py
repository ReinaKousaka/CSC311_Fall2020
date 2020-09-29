''' Author: Songheng Yin,
    Contact: songheng.yin@mail.utoronto.ca'''

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


def load_data():
    text = []
    # Extract all sentences in clean_fake.txt
    num_fakelines = 0
    with open('./data/clean_fake.txt') as file:
        lines = file.readlines()
        for line in lines:
            text.append(line)
            num_fakelines += 1
    # Extract all sentences in clean_real.txt
    num_reallines = 0
    with open('./data/clean_real.txt') as file:
        lines = file.readlines()
        for line in lines:
            text.append(line)
            num_reallines += 1
    # Vectorizer
    vectorizer = CountVectorizer()
    input_matrix = vectorizer.fit_transform(text)
    targets = np.concatenate((np.zeros(num_fakelines), np.ones(num_reallines)), axis=0)  # 0: Fake 1: Real
    # Spilt our datasets into 70%, 15%, 15%
    X_train, X_test, y_train, y_test = train_test_split(input_matrix, targets, test_size=0.3)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def select_knn_model(X_train, X_valid, X_test, y_train, y_valid, y_test, metric='minkowski'):  # default metric is minkowskil
    train_accuracy_list, valid_accuracy_list = [], []
    for k in range(1, 21):
        # Build our KNN model
        knn_model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn_model.fit(X_train, y_train)
        # Use the model predict result on our training set 
        train_pred = knn_model.predict(X_train)
        train_accuracy = 1 - (np.sum(np.abs(train_pred - y_train)) / len(train_pred))
        train_accuracy_list.append(train_accuracy)
        # Use the model predict result on our validation set 
        valid_pred = knn_model.predict(X_valid)
        valid_accuracy = 1 - (np.sum(np.abs(valid_pred - y_valid)) / len(valid_pred))
        valid_accuracy_list.append(valid_accuracy)
    # Choose the best model based on validation accuracy
    best_k = np.argmax(valid_accuracy_list) + 1
    # Similarly, build the best model
    knn_model = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    knn_model.fit(X_train, y_train)
    # Use it to predict on the test set
    test_pred = knn_model.predict(X_test)
    test_accuracy = 1 - (np.sum(np.abs(test_pred - y_test)) / len(test_pred))
    print('The k value of the best model is: {}. Its accuracy on the test set is {}.'.format(best_k, test_accuracy))
    return train_accuracy_list, valid_accuracy_list
    
def _draw(train_accuracy_list, valid_accuracy_list):
    """ The helper function that draws the graph of accuracy"""
    fig, ax = plt.subplots(1, 1)
    x_axis = np.arange(1, 21)
    ax.plot(x_axis, train_accuracy_list, label='Training')
    ax.plot(x_axis, valid_accuracy_list, label='Validation')
    ax.set_xlabel('K value')
    ax.set_ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Question(a)
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    
    # Question(b)
    train_accuracy_list, valid_accuracy_list= select_knn_model(X_train, X_valid, X_test, y_train, y_valid, y_test)
    _draw(train_accuracy_list, valid_accuracy_list)
    
    # Question(c)
    # train_accuracy_list, valid_accuracy_list= select_knn_model(X_train, X_valid, X_test, y_train, y_valid, y_test, metric='cosine')
    # _draw(train_accuracy_list, valid_accuracy_list)