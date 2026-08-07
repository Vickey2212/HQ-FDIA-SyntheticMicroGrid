class EmbeddingDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.X = data["X"].float()
        self.y = data["y"].long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
