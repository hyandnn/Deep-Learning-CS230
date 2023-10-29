# coding: utf-8

import torch


class Model(torch.nn.Module):
    def __init__(self, N_in, h1, h2, D_out):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(N_in, h1)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(h1, h2)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(h2, D_out)
        self.softmax = torch.nn.Softmax(dim=1)
        self.model = torch.nn.Sequential(self.linear1, self.relu1, self.linear2, self.relu2, self.linear3, self.softmax)

    # def forward(self, x):
    #     linear1 = self.linear1(x)
    #     relu1 = self.relu1(linear1)
    #
    #     linear2 = self.linear2(relu1)
    #     relu2 = self.relu2(linear2)
    #
    #     linear3 = self.linear3(relu2)
    #     y_pred = self.softmax(linear3)
    #     return y_pred
    def forward(self, x):
        return self.model(x)
