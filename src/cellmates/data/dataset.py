from torch.utils.data import Dataset
from cellmates.data.sample import Sample


class CellMatesDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
