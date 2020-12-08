import sys
sys.path.append('..')
from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier

def knn_impute_by_user(matrix, valid_data, k, print_option=True):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    if print_option:
        print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    # train_data = load_train_csv('../data')
    # print(len(train_data['question_id']))
    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Question 1(a)
    acc_list_user, k_values = [], [1, 6, 11, 16, 21, 26]
    for k in k_values:
        acc_list_user.append(knn_impute_by_user(sparse_matrix, val_data, k))
    fig, ax = plt.subplots(1, 1)
    ax.plot(k_values, acc_list_user)
    ax.set_xlabel('K value')
    ax.set_ylabel('Accuracy on validation data')
    ax.set_title('PartA 1(a)')
    plt.savefig('./1(a).png')

    # Question 2(a)
    best_k_user = k_values[np.argmax(acc_list_user)]
    test_acc_user = knn_impute_by_user(sparse_matrix, test_data, best_k_user, print_option=False)
    print('User-based: Chosen k = {}, Test accuracy = {}.'.format(best_k_user, test_acc_user))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
