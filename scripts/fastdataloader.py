import numpy as np
from torch.utils.data.dataloader import DataLoader


class FastDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuggle = shuffle
        self.drop_last = drop_last
        self.batch_idx_last = []
        self.len = len(self.dataset) // self.batch_size
        if not drop_last and len(self.dataset) % self.batch_size > 0:
            self.len += 1

        idx = np.arange(len(self.dataset))
        if shuffle:
            np.random.shuffle(idx)
        last = len(idx) % batch_size
        if last > 0:
            self.batch_idx_last = idx[-last:]
            idx = idx[:-last]
        idx = idx.reshape(-1, batch_size)
        self.batch_idx = idx

    def __iter__(self):
        for idx in self.batch_idx:
            yield self.dataset[idx]
        if len(self.batch_idx_last) > 0 and not self.drop_last:
            yield self.dataset[self.batch_idx_last]

    def __len__(self):
        return self.len
