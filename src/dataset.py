import scanpy as sc
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings_array):
        self.embeddings = torch.tensor(embeddings_array, dtype=torch.float32)

    def __len__(self):
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        return self.embeddings[idx]