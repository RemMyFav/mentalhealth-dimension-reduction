import torch
from torch.utils.data import Dataset
import numpy as np


class RankDataset(Dataset):
    """
    Dataset for ranking model.

    X: sentence embeddings
        shape = [N, embedding_dim]

    Y: target rank labels
        shape = [N, num_labels]
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]