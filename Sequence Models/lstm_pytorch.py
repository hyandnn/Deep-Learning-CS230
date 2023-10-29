# coding: utf-8

import torch


class LSTM_EMO(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LSTM_EMO, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.5, batch_first=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(128, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(h_n[-1])
        linear_out = self.fc(out)
        return linear_out

    def predict(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(h_n[-1])
        linear_out = self.fc(out)
        y_pre = self.softmax(linear_out)
        return y_pre






