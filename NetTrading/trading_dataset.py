import torch
import torch.utils.data as data

class NetGirlDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.x_train = torch.FloatTensor(
            [(-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1), (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)])
        self.y_train = torch.FloatTensor([-1, 1, -1, 1, -1, 1, -1, -1])

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

    def __len__(self):
        return len(self.y_train)