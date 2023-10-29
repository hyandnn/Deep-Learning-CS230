# coding: utf-8

import torch


class CNN_digit(torch.nn.Module):
    def __init__(self):
        super(CNN_digit, self).__init__()
        self.conv2d1 = torch.nn.Conv2d(in_channels=3, out_channels=8, stride=1, padding=1, kernel_size=4)
        self.relu1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=8, stride=8)

        self.conv2d2 = torch.nn.Conv2d(in_channels=8, out_channels=16, stride=1, padding=0, kernel_size=2)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=4, stride=4)

        self.fc = torch.nn.Linear(16, 6)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        conv2d1 = self.conv2d1(x)
        relu1 = self.relu1(conv2d1)
        maxpool1 = self.maxpool1(relu1)

        conv2d2 = self.conv2d2(maxpool1)
        relu2 = self.relu2(conv2d2)
        maxpool2 = self.maxpool2(relu2)

        batch_size = maxpool2.size()[0]
        maxpool2 = maxpool2.view(batch_size, -1)

        fc = self.fc(maxpool2)

        return fc

    def test(self, x):
        y_pred = self.forward(x)
        y_predict = self.softmax(y_pred)
        return y_predict
