####
#### Data classes for use with pytorch dataloader

import numpy as np
from torch.utils import data

class asc_dynamic(data.Dataset):
    
    def __init__(self,dry,wet,width):
        self.dry = np.hstack((np.zeros(width),dry))
        self.wet = wet
        self.width = width

    def __len__(self):
        return len(self.wet)

    def __getitem__(self, idx):
        dry = self.dry[idx:idx+self.width]
        wet = self.wet[idx]
        return dry, wet