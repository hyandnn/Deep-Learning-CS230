# coding: utf-8

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from cnn_model import CNN_digit
from data_utils import Digit_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

train_data_path = "datasets/train_signs.h5"
test_data_path = "datasets/test_signs.h5"

if __name__ == "__main__":
    # define some parameters
    num_epoch = 200
    learning_rate = 0.00095
    minibatch_size = 32
    costs = []

    # load data
    train_data = Digit_data(train_data_path)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=minibatch_size)

    # Initialize the model, initialize the loss function, define the optimizer
    m = CNN_digit()
    m.to(device)  # GPU speed up

    loss_fn = torch.nn.CrossEntropyLoss()  # cross-entropy

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)  # 使用Adam优化算法

    # training
    for epoch in range(num_epoch):
        cost = 0
        for i, data in enumerate(train_data_loader):
            img_data, img_label = data
            img_data = img_data.permute(0, 3, 1, 2)  # 将维度从（32，64，64，3）转换为（32，3，64，64）
            img_data = img_data.to(device)
            img_label = img_label.to(device)

            optimizer.zero_grad()

            y_pred = m.forward(img_data)

            loss = loss_fn(y_pred, img_label.long())

            loss.backward()

            optimizer.step()

            cost = cost + loss.cpu().detach().numpy()
        costs.append(cost / (i + 1))
        if epoch % 5 == 0:
            print("epoch=" + str(epoch) + ":  " + "loss=" + str(cost / (i + 1)))

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # calculate acc
    test_data = Digit_data(test_data_path)
    test_data_loader = DataLoader(test_data, shuffle=True, batch_size=minibatch_size)

    acc_train = 0
    acc_test = 0
    correct_train = torch.zeros(1).squeeze().cuda()
    total_train = torch.zeros(1).squeeze().cuda()
    for i, data in enumerate(train_data_loader):
        img_data, img_label = data
        img_data = img_data.permute(0, 3, 1, 2)

        img_data = img_data.to(device)
        img_label = img_label.to(device)

        pred = m.test(img_data)

        prediciton = torch.argmax(pred, dim=1)

        correct_train += (prediciton == img_label).sum().float()
        total_train += len(img_label)

    acc_train = (correct_train / total_train).cpu().detach().data.numpy()

    print("Acc on the train set is："+str(acc_train))

    correct_test = torch.zeros(1).squeeze().cuda()
    total_test = torch.zeros(1).squeeze().cuda()
    for j, data in enumerate(test_data_loader):
        img_data, img_label = data
        img_data = img_data.permute(0, 3, 1, 2)

        img_data = img_data.to(device)
        img_label = img_label.to(device)

        pred = m.test(img_data)

        prediciton = torch.argmax(pred, dim=1)

        correct_test += (prediciton == img_label).sum().float()
        total_test += len(img_label)

    acc_test = (correct_test / total_test).cpu().detach().data.numpy()
    print("Acc on the test set is：" + str(acc_test))





