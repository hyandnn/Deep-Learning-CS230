# coding: utf-8

import numpy as np
import w2v_utils_pytorch

import emo_utils

X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')

max_len = len(max(X_train, key=len).split())

index = 3
print(X_train[index], emo_utils.label_to_emoji(Y_train[index]))


def sentence_to_avg(sentence, word_to_vec_map):
    """
    :param sentence
    :param word_to_vec_map
    :return:
    """
    # seq to words
    words = sentence.lower().split()

    # init average vector
    avg = np.zeros(50, )

    for w in words:
        avg = avg + word_to_vec_map[w]
    avg = np.divide(avg, len(words))
    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    train model
    :param X:
    :param Y:
    :param word_to_vec_map:
    :param learning_rate:
    :param num_iterations:
    :return:
    """
    np.random.seed(1)

    m = Y.shape[0]
    n_y = 5
    n_h = 50

    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)

    for epoch in range(num_iterations):
        for i in range(m):
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # forward
            z = np.dot(W, avg) + b
            a = emo_utils.softmax(z)

            # cal ith loss
            cost = -np.sum(Y_oh[i] * np.log(a))

            # gradient
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # update
            W = W - learning_rate * dW
            b = b - learning_rate * db
        if epoch % 100 == 0:
            print("epoch is {epoch}，loss is {cost}".format(epoch=epoch, cost=cost))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)
    return pred, W, b


if __name__ == "__main__":
    words, word_to_vec_map = w2v_utils_pytorch.read_glove_vecs('data/glove.6B.50d.txt')
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    print("=====trainset====")
    pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
    print("=====testset====")
    pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)
    X_my_sentences = np.array(
        ["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
    Y_my_labels = np.array([[0], [0], [2], [1], [4], [3]])

    pred = emo_utils.predict(X_my_sentences, Y_my_labels, W, b, word_to_vec_map)
    emo_utils.print_predictions(X_my_sentences, pred)

