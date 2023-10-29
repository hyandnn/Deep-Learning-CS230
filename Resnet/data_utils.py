# coding: utf-8

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class Image_Data(Dataset):
    def __init__(self, data_path):
        super(Image_Data, self).__init__()
        # 读取数据集
        dataset = h5py.File(data_path, "r")
        if data_path == "datasets/train_happy.h5" or data_path == "datasets/train_signs.h5":
            data_set_x_orig = np.array(dataset["train_set_x"][:])
            data_set_y_orig = np.array(dataset["train_set_y"][:])
        else:
            data_set_x_orig = np.array(dataset["test_set_x"][:])
            data_set_y_orig = np.array(dataset["test_set_y"][:])

        data_set_x_orig = data_set_x_orig.astype("float32") / 255
        data_set_y_orig = data_set_y_orig.astype("float32")

        self.x_data = torch.from_numpy(data_set_x_orig)
        self.y_data = torch.from_numpy(data_set_y_orig)

        self.len = self.y_data.size()[0]

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len

    def get_shape(self):
        return self.x_data.size(), self.y_data.size()


if __name__ == "__main__":
    happy_data = Image_Data("datasets/train_signs.h5")
    train_data_loader = DataLoader(dataset=happy_data, batch_size=32, shuffle=True)
    for i, data in enumerate(train_data_loader):
        img_data, img_label = data
        # print(img_label)
        plt.imshow(img_data[23])
        plt.show()
    print(happy_data.get_shape())
