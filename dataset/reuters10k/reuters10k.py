import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset

current_file_path = os.path.abspath(__file__)
dataset_path = os.path.abspath(os.path.dirname(current_file_path))

class Reuters10kDataset(Dataset):
    '''Reuters 10k dataset torchvision wrapper
        Args:
            data_path: path/to/save/the/reuters10k.mat (filename not included)
            train: if True split 80% of total dataset as train dataset otherwise 20% as test set
        Outputs:
            data: shape (batch_size, 2000) text tokens
            labels: shape (batch_size, 1) indicators of text categories
    '''
    def __init__(self, train=True):
        # load data and labels from data_path
        self.content=scio.loadmat(dataset_path+'/reuters10k.mat')
        length = len(self.content['X'])
        train_limits = int(0.8*length)
        self.targets = self.content['Y'].squeeze()

        if train is True: # split train/test == 8/2
            self.data = self.content['X'][:train_limits]
            self.targets = self.targets[:train_limits]
        else:
            self.data = self.content['X'][train_limits:]
            self.targets = self.targets[train_limits:]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # get data and label at idx
        x = self.data[idx]
        y = self.targets[idx]
        # convert data to tensor and normalize
        x = torch.tensor(x, dtype=torch.float32)
        # return data and label as tuple
        return x, y