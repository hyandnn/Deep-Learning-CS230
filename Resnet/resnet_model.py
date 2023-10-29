# coding: utf-8

import torch


class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Residual, self).__init__()
        self.stride = stride
        # The convolutional layer is filled with same,
        # and since pytorch does not provide an auto-fill operation,
        # you need to calculate the size of the fill by hand
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.conv1x1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        out1 = self.relu1(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))

        if self.conv1x1:
            x = self.conv1x1(x)

        out = self.relu1(out2 + x)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 64)
        )

        self.conv3 = torch.nn.Sequential(
            Residual(64, 128, stride=2),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128),
            Residual(128, 128)
        )

        self.conv4 = torch.nn.Sequential(
            Residual(128, 256, stride=2),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256),
            Residual(256, 256)
        )

        self.conv5 = torch.nn.Sequential(
            Residual(256, 512, stride=2),
            Residual(512, 512),
            Residual(512, 512)
        )

        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)

        out = self.fc(out)
        return out
