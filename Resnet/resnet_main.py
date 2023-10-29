# coding: utf-8

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from data_utils import Image_Data
from resnet_model import ResNet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

train_happy_data_path = "datasets/train_happy.h5"
train_signs_data_path = "datasets/train_signs.h5"

test_happy_data_path = "datasets/test_happy.h5"
test_signs_data_path = "datasets/test_signs.h5"

if __name__ == "__main__":
    num_epoch = 50
    learning_rate = 0.01
    batch_size = 32
    costs = []
    classes = 6

    train_data = Image_Data(train_signs_data_path)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    test_data = Image_Data(test_signs_data_path)
    test_data_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    m = ResNet(3, num_classes=classes)
    m = m.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    for epoch in range(num_epoch):
        cost = 0
        for i, data in enumerate(train_data_loader):
            img_data, img_label = data
            img_data = img_data.permute(0, 3, 1, 2)

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

    correct_train = torch.zeros(1).squeeze().cuda()
    total_train = torch.zeros(1).squeeze().cuda()
    for i, data in enumerate(train_data_loader):
        img_data, img_label = data
        img_data = img_data.permute(0, 3, 1, 2)

        img_data = img_data.to(device)
        img_label = img_label.to(device)

        y_pred = m.forward(img_data)

        y_pred = torch.nn.Softmax(dim=1)(y_pred)

        prediciton = torch.argmax(y_pred, dim=1)

        correct_train += (prediciton == img_label).sum().float()
        total_train += len(img_label)

    acc_train = (correct_train / total_train).cpu().detach().data.numpy()

    print("Acc on the train set is：" + str(acc_train))

    correct_test = torch.zeros(1).squeeze().cuda()
    total_test = torch.zeros(1).squeeze().cuda()
    for j, data in enumerate(test_data_loader):
        img_data, img_label = data
        img_data = img_data.permute(0, 3, 1, 2)

        img_data = img_data.to(device)
        img_label = img_label.to(device)

        y_pred = m.forward(img_data)

        y_pred = torch.nn.Softmax(dim=1)(y_pred)

        prediciton = torch.argmax(y_pred, dim=1)

        correct_test += (prediciton == img_label).sum().float()
        total_test += len(img_label)

    acc_test = (correct_test / total_test).cpu().detach().data.numpy()
    print("Acc on th test set is：" + str(acc_test))



