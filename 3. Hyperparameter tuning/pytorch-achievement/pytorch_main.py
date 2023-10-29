# coding: utf-8

import torch

num = torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot
from model import Model


def data_processing():
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T / 255
    X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T / 255

    # Y_train = convert_to_one_hot(Y_train_orig, 6)
    # Y_test = convert_to_one_hot(Y_test_orig, 6)

    return X_train_flatten, Y_train_orig, X_test_flatten, Y_test_orig, classes


if __name__ == "__main__":
    X_train_flatten, Y_train, X_test_flatten, Y_test, classes = data_processing()

    X_train_flatten = torch.from_numpy(X_train_flatten).to(torch.float32).to(device)
    Y_train = torch.from_numpy(Y_train).to(torch.float32).to(device)
    X_test_flatten = torch.from_numpy(X_test_flatten).to(torch.float32).to(device)
    Y_test = torch.from_numpy(Y_test).to(torch.float32).to(device)

    D_in, h1, h2, D_out = 12288, 25, 12, 6
    m = Model(D_in, h1, h2, D_out)
    m.to(device)
    epoch_num = 1500
    learning_rate = 0.0001
    minibatch_size = 32
    seed = 3
    costs = []
    optimizer = torch.optim.Adam(m.model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        epoch_cost = 0
        num_minibatches = int(X_train_flatten.size()[1] / minibatch_size)
        minibatches = random_mini_batches(X_train_flatten, Y_train, minibatch_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            y_pred = m.forward(minibatch_X.T)

            y = minibatch_Y.T

            y = y.view(-1)

            loss = loss_fn(y_pred, y.long())

            epoch_cost = epoch_cost + loss.item()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
        epoch_cost = epoch_cost / (num_minibatches + 1)
        if epoch % 5 == 0:
            costs.append(epoch_cost)
            # 是否打印：
            if epoch % 100 == 0:
                print("epoch = " + str(epoch) + "    epoch_cost = " + str(epoch_cost))
