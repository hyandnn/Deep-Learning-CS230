# coding: utf-8

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import h5py
import matplotlib.pyplot as plt


class Digit_data(Dataset):
    def __init__(self, data_path):
        super(Digit_data, self).__init__()
        dataset = h5py.File(data_path, "r")
        if data_path == "datasets/train_signs.h5":
            dataset_set_x_orig = np.array(dataset["train_set_x"][:])  # your train set features
            dataset_set_y_orig = np.array(dataset["train_set_y"][:])  # your train set labels
        else:
            dataset_set_x_orig = np.array(dataset["test_set_x"][:])  # your train set features
            dataset_set_y_orig = np.array(dataset["test_set_y"][:])  # your train set labels

        dataset_set_x_orig = dataset_set_x_orig.astype("float32") / 255
        dataset_set_y_orig = dataset_set_y_orig.astype("float32")

        self.x_data = torch.from_numpy(dataset_set_x_orig)
        self.y_data = torch.from_numpy(dataset_set_y_orig)

        self.len = self.y_data.size()[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

# cat_data = Digit_data('datasets/train_signs.h5')
#
# train_data_loader = DataLoader(dataset=cat_data,
#                                batch_size=32,
#                                shuffle=True)
# for i, data in enumerate(train_data_loader):
#     inputs, labels = data
#     plt.imshow(inputs[23])
#     plt.show()
#     print("inputs", inputs.data.size(), "labels", labels.data.size())
