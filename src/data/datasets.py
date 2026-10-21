import torch
from torch.utils.data import DataLoader, Dataset, random_split

class PointCloudDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        if X.dim() != 2:
            raise ValueError(f"X must have shape (N, d), got {X.shape}")
        if y.dim() != 1:
            raise ValueError(f"y must have shape (N,), got {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length, got {len(X)} and {len(y)}")

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_datasets(X, y,  train_ratio=0.8, seed=42):
    dataset = PointCloudDataset(X, y)

    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_test = n_total - n_train

    train_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    return train_dataset, test_dataset