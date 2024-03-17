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
    """
    Collate samples.

        - Pad cell_types and distances with zeros
        - Initialize a padding matrix of shape BL with
          ones up to the number of cells of each sample.
    """
    # batch dimensions:
    B = len(samples)
    L = max([s.n_cells for s in samples])

    # initialize batch with zero tensors:
    batch = {
        "cell_types_BL": torch.zeros((B, L), dtype=int),
        "distances_BLL": torch.zeros((B, L, L), dtype=int),
        "responder_cell_type_B": torch.zeros(B, dtype=int),
        # TODO make 32 or 64 a global config:
        "is_dividing_B": torch.zeros(B, dtype=torch.float32),
        "padding_mask_BL": torch.zeros((B, L), dtype=torch.float32),
    }

    # write samples into the zero tensors, leaving zeros as padding:
    for i, sample in enumerate(samples):

        # sample sequence length
        _L = sample.n_cells

        # write:
        batch["cell_types_BL"][i, :_L] = torch.Tensor(sample.cell_types)
        batch["distances_BLL"][i, :_L, :_L] = torch.Tensor(sample.distances)
        batch["responder_cell_type_B"][i] = sample.responder_cell_type
        batch["is_dividing_B"][i] = sample.is_dividing
        batch["padding_mask_BL"][i, :_L] = torch.ones(_L)

    return batch
