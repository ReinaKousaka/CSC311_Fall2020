# TODO: complete this file.
from part_a.item_response import *


def bootstrap(data):
    """
    return a bootstrap sample of data with the same sample size as data.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """

    usr = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])

    resample = np.random.choice(len(usr), len(usr), replace=True)

    return {"user_id": usr[resample],
            "question_id": q[resample],
            "is_correct": c[resample]}

def bootstrap_evaluate(data, theta_ls, beta_ls):
    """ Generate predictions using the trained models with bootstrap,
    average the predictions and return the general accuracy.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta_ls: a list of Vectors trained by bootstrap
    :param beta_ls: a list of Vectors trained by bootstrap
    :return: float
    """

    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        bootstrap_pred = []
        for j in range(len(theta_ls)):
            x = (theta_ls[j][u] - beta_ls[j][q]).sum()
            p = sigmoid(x)
            bootstrap_pred.append(p)
        pred.append(np.mean(bootstrap_pred) >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


if __name__ == '__main__':
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Train the first model with bootstrap:")
    theta_1, beta_1, val_acc_1, train_ll_1, valid_ll_1 \
        = irt(bootstrap(train_data), val_data, 0.01, 50)

    print("Train the second model with bootstrap:")
    theta_2, beta_2, val_acc_2, train_ll_2, valid_ll_2 \
        = irt(bootstrap(train_data), val_data, 0.01, 50)

    print("Train the third model with bootstrap:")
    theta_3, beta_3, val_acc_3, train_ll_3, valid_ll_3 \
        = irt(bootstrap(train_data), val_data, 0.01, 50)

    # evaluate each single base model
    valid_accuracy_1 = evaluate(val_data, theta_1, beta_1)
    valid_accuracy_2 = evaluate(val_data, theta_2, beta_2)
    valid_accuracy_3 = evaluate(val_data, theta_3, beta_3)
    test_accuracy_1 = evaluate(test_data, theta_1, beta_1)
    test_accuracy_2 = evaluate(test_data, theta_2, beta_2)
    test_accuracy_3 = evaluate(test_data, theta_3, beta_3)
    print("Validation accuracy for each single base model: ")
    print([valid_accuracy_1, valid_accuracy_2, valid_accuracy_3])
    print("Test accuracy for each single base model: ")
    print([test_accuracy_1, test_accuracy_2, test_accuracy_3])

    # evaluate the bootstrap models by average their predictions
    theta_ls = [theta_1, theta_2, theta_3]
    beta_ls = [beta_1, beta_2, beta_3]
    final_val_accuracy = bootstrap_evaluate(val_data, theta_ls, beta_ls)
    final_test_accuracy = bootstrap_evaluate(test_data, theta_ls, beta_ls)
    print("Ensemble validation accuracy is: {}".format(final_val_accuracy))
    print("Ensemble test accuracy is: {}".format(final_test_accuracy))
