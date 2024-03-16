from torch.utils.data import Dataset
from cellmates.data.sample import Sample
import torch


class CellMatesDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def save(self, path):
        torch.save(self, path)

    def load(path):
        return torch.load(path)


def collate_fn(samples):
    # TODO: implement the collate_fn function for batching samples in different sizes

    # if cell_types of each sample is in different length
    # then we can't use torch.stack to make a tensor
    # so we need to pad the cell_types to the same length
    # max_len = max([len(s.cell_types) for s in samples])
    # for s in samples:
    #     s.cell_types += [0.0] * (max_len - len(s.cell_types))

    # # we also need to pad the distances to the same shape
    # max_len = max([s.distances.shape[0] for s in samples])
    # for s in samples:
    #     s.distances = torch.cat(
    #         [
    #             s.distances,
    #             torch.zeros(max_len - s.distances.shape[0], s.distances.shape[1]),
    #         ],
    #         dim=0,
    #     )
    #     s.distances = torch.cat(
    #         [
    #             s.distances,
    #             torch.zeros(s.distances.shape[0], max_len - s.distances.shape[1]),
    #         ],
    #         dim=1,
    #     )

    # stack the cell_types, distances and targets

    cell_types_BL = torch.stack(
        [torch.tensor(s.cell_types, dtype=torch.long) for s in samples]
    )
    distances_BLL = torch.stack([s.distances for s in samples])
    targets = torch.stack(
        [torch.tensor(float(s.is_dividing)).unsqueeze(-1) for s in samples]
    )

    return cell_types_BL, distances_BLL, targets
