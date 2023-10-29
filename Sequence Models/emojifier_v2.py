# coding: utf-8

from torch.utils.data import Dataset, DataLoader
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from data_utils import Sentence_Data
import lstm_pytorch
import matplotlib.pyplot as plt

train_data_path = "data/train_emoji.csv"
test_data_path = "data/test.csv"


if __name__ == "__main__":
    # init parameters
    batch_size = 32
    epoch_nums = 1000
    learning_rate = 0.001
    costs = []
    input_size = 50
    num_classes = 5
    # load train set
    train_data = Sentence_Data(train_data_path)
    train_data_loader = DataLoader(train_data, shuffle=True, batch_size=32)

    # init model
    m = lstm_pytorch.LSTM_EMO(input_size=input_size, num_classes=num_classes)
    m.to(device)

    # define loss func and opti
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

    # training
    print("learning_rate=" + str(learning_rate))
    for epoch in range(epoch_nums):
        cost = 0
        index = 0
        for data, label in train_data_loader:
            data, label = data.to(device), label.to(device)

            optimizer.zero_grad()

            y_pred = m.forward(data)

            loss = loss_fn(y_pred, label.long())
            loss.backward()

            optimizer.step()

            cost = cost + loss.cpu().detach().numpy()
            index = index + 1
        if epoch % 50 == 0:
            costs.append(cost / index)
            print("epoch=" + str(epoch) + ":  " + "loss=" + str(cost / (index + 1)))

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    # testing
    acc_train = 0
    acc_test = 0
    correct_train = torch.zeros(1).squeeze().cuda()
    total_train = torch.zeros(1).squeeze().cuda()

    test_data = Sentence_Data(test_data_path)
    test_data_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    for data, label in train_data_loader:
        data, label = data.to(device), label.to(device)
        y_pred = m.predict(data)
        prediction = torch.argmax(y_pred, dim=1)
        correct_train += (prediction == label).sum().float()
        total_train += len(label)
    acc_train = (correct_train / total_train).cpu().detach().data.numpy()
    print("acc on the train set is：" + str(acc_train))


    correct_test = torch.zeros(1).squeeze().cuda()
    total_test = torch.zeros(1).squeeze().cuda()
    for data, label in test_data_loader:
        data, label = data.to(device), label.to(device)
        y_pred = m.predict(data)
        prediction = torch.argmax(y_pred, dim=1)
        correct_test += (prediction == label).sum().float()
        total_test += len(label)
    acc_test = (correct_test / total_test).cpu().detach().data.numpy()
    print("acc on the test set is：" + str(acc_test))






