import numpy as np

class NextCharDataset:
    def __init__(self, data, seq_length):
        self.data = data.copy()

        self.window_view = np.lib.stride_tricks.sliding_window_view(
            self.data, window_shape=seq_length + 1
        )

    def __len__(self):
        return len(self.window_view)

    def __getitem__(self, idx):
        x, y = self.window_view[idx, :-1], self.window_view[idx, 1:]
        return x, y

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=False, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        if self.drop_last:
            remainder = len(self.dataset) % self.batch_size
            if remainder:
                indices = indices[:-remainder]

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.dataset[j] for j in batch_indices]
            yield self.collate_fn(batch)
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return np.ceil(len(self.dataset) / self.batch_size).astype(int)

    def collate_fn(self, batch):
        if isinstance(batch[0], (tuple, list)):
            return [np.array(samples) for samples in zip(*batch)]
        elif isinstance(batch[0], dict):
            return {
                key: np.array([d[key] for d in batch]) for key in batch[0]
            }
        else:
            return np.array(batch)
