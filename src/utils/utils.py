# -*- coding: utf-8 -*-
import os
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset
import configparser


def ReadConfig(configfile):
    config = configparser.ConfigParser()
    config.read(configfile)
    cfgPath = config['path']
    cfgTrain = config['train']
    cfgModel = config['model']
    return cfgPath, cfgTrain, cfgModel

class ISRUCS3(Dataset):

    def __init__(self, paths=None, transforms=None) -> None:
        """
        """
        super().__init__()
        self.paths = paths
        self.transform = transforms

    def __getitem__(self, index):
        filePath = self.paths[index]
        saved_data = loadmat(filePath)
        data = saved_data['data'].transpose(1, 0)
        label = int(filePath.split(os.sep)[-1].split('_')[-1].split('.')[0])

        if self.transform is not None:
            data = self.transform(data)

        if data.ndim == 3:
            data = data.transpose(0, 2, 1)
            label = np.stack((label, label), 0)
        else:
            data = data.transpose(1, 0)

        return data, label

    def __len__(self):
        return len(self.paths)