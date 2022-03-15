import pickle
import os
import torch.utils.data
import numpy as np
import h5py

from tqdm import tqdm
from torch.utils.data import DataLoader


class Dataset_F3(torch.utils.data.Dataset):
    """
    build the dataset for the F3 model
    """
    def __init__(self, mode):
        self.dataset = []
        file_path = f'./data/{mode}'
        file_list = os.listdir(file_path)
        for file in file_list:
            data_path = f'{file_path}/{file}'
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.dataset.append(data)
        self.data_len = len(self.dataset)

    def __getitem__(self, index):
        """
        provide data items for model training
        :param index:
        :return:
        """
        return self.dataset[index]

    def __len__(self):
        return self.data_len


def test():
    dataset = Dataset_F3('valid')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for _, batch in enumerate(dataloader):
        print(batch)


if __name__ == '__main__':
    test()
