from utils import *
from part_a.item_response import *

def load_question_meta():
    # load question_meta.csv
    path = os.path.join("../data", "question_meta.csv")
    if not os.path.exists(path):
        raise Exception("The specified path {} does not exist.".format(path))
    # Initialize the data.
    data = {
        "question_id": [],
        "subject_id": []
    }
    # Iterate over the row to fill in the data.
    with open(path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                data["question_id"].append(int(row[0]))
                s = str(row[1])[1:-1]
                subjects = s.split(", ")
                subjects = [int(x) - 1 for x in subjects]
                data["subject_id"].append(subjects)
            except ValueError:
                # Pass first row.
                pass
            except IndexError:
                # is_correct might not be available.
                pass
    return data

def meta_to_matrix(data):
    """
    Transform the question_meta data into matrix representation
    """

    q = data["question_id"]
    sub = data["subject_id"]

    result = np.zeros((1774, 387))
    for i in range(len(q)):
        result[q[i], sub[i]] = 1
    return result


def neg_log_likelihood_b(data, subject, theta, beta, gamma):
    """ Compute the negative log-likelihood.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param subject: Matrix
    :param theta: Vector
    :param beta: Vector
    :param gamma: Vector
    :return: float
    """

    usr = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    para = theta[usr] - beta[q] - subject[q].dot(gamma)
    log_like = np.log(sigmoid(para)) * c + np.log(1 - sigmoid(para)) * (1 - c)
    log_lklihood = np.sum(log_like)

    return -log_lklihood

def update_b(data, subject, lr, theta, beta, gamma):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param subject: Matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param gamma: Vector
    :return: tuple of vectors
    """

    usr = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    para = theta[usr] - beta[q] - subject[q].dot(gamma)
    values = c - sigmoid(para)

    sparse = csc_matrix((values, (usr, q)), shape=(len(theta), len(beta))).toarray()
    theta = theta + np.sum(sparse, axis=1) * lr

    para = theta[usr] - beta[q] - subject[q].dot(gamma)
    values = c - sigmoid(para)
    sparse = csc_matrix((values, (usr, q)), shape=(len(theta), len(beta))).toarray()
    for i in range(len(gamma)):
        grad = np.sum(sparse.dot(subject[:, i]))
        gamma[i] = gamma[i] - grad * lr * 0.012

    para = theta[usr] - beta[q] - subject[q].dot(gamma)
    values = c - sigmoid(para)
    sparse = csc_matrix((values, (usr, q)), shape=(len(theta), len(beta))).toarray()
    beta = beta - np.sum(sparse, axis=0) * lr

    return theta, beta, gamma

def irt_b(data, subject, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param subject: Matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, gamma, val_acc_lst)
    """

    theta = np.zeros(542)
    beta = np.zeros(1774)
    gamma = np.zeros(387)

    val_acc_lst = []
    train_loglik = []
    valid_loglik = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood_b(data, subject, theta, beta, gamma)
        val_neg_lld = neg_log_likelihood_b(val_data, subject, theta, beta, gamma)
        train_loglik.append(-neg_lld)
        valid_loglik.append(-val_neg_lld)

        score = evaluate_b(val_data, subject, theta, beta, gamma)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, gamma = update_b(data, subject, lr, theta, beta, gamma)

    return theta, beta, gamma, val_acc_lst, train_loglik, valid_loglik


def evaluate_b(data, subject, theta, beta, gamma):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param subject: Matrix
    :param theta: Vector
    :param beta: Vector
    :param gamma: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q] - subject[q].dot(gamma)).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


if __name__ == '__main__':
    subject_meta = load_question_meta()
    subject_matrix = meta_to_matrix(subject_meta)

    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Train baseline model")
    theta, beta, val_acc, train_ll, valid_ll \
        = irt(train_data, val_data, 0.01, 70)
    print("Train modified model")
    theta_b, beta_b, gamma_b, val_acc_b, train_ll_b, valid_ll_b \
        = irt_b(train_data, subject_matrix, val_data, 0.01, 70)

    validation_accuracy = evaluate(val_data, theta, beta)
    test_accuracy = evaluate(test_data, theta, beta)
    print("Baseline validation accuracy is: " + str(validation_accuracy))
    print("Baseline test accuracy is: " + str(test_accuracy))

    validation_accuracy = evaluate_b(val_data, subject_matrix, theta_b, beta_b, gamma_b)
    test_accuracy = evaluate_b(test_data, subject_matrix, theta_b, beta_b, gamma_b)
    print("Modified validation accuracy is: " + str(validation_accuracy))
    print("Modified test accuracy is: " + str(test_accuracy))

    plt.plot(train_ll, label="Baseline Training Log-likelihood")

    plt.plot(train_ll_b, label="Modified training Log-likelihood")

    plt.xlabel("Num of Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood Comparison")
    plt.legend()
    plt.show()

    plt.plot(valid_ll, label="Baseline Validation Log-likelihood")
    plt.plot(valid_ll_b, label="Modified validation Log-likelihood")
    plt.xlabel("Num of Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood Comparison")
    plt.legend()
    plt.show()

    plt.plot(val_acc, label="Baseline Validation Accuracy")
    plt.plot(val_acc_b, label="Modified Validation Accuracy")
    plt.xlabel("Num of Iteration")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.show()
