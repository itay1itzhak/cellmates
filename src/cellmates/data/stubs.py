from cellmates.utils import microns
from cellmates.data.sample import Sample
from cellmates.data.dataset import CellMatesDataset

import torch
import numpy as np


def generate_dataset_for_distances() -> CellMatesDataset:
    """
    A dataset of 15 samples including 2 cells, the distance of the second cell determines division.
    """

    def second_cell_distance(second_cell_distance: float) -> Sample:
        return Sample(
            cell_types=[1, 1],
            distances=torch.tensor(
                [[0, second_cell_distance], [second_cell_distance, 0]]
            ),
            responder_cell_type=1,
            is_dividing=microns(second_cell_distance) < microns(40),
        )

    samples = [second_cell_distance(d) for d in range(0, 150, 10)]
    return CellMatesDataset(samples)


def generate_dataset_for_cell_type() -> CellMatesDataset:
    """
    Generates a dataset with 5 samples, with two cells - the second cell type varies and determines division.
    """

    def second_cell_type_sample(second_cell_type: int) -> Sample:
        return Sample(
            cell_types=[1, second_cell_type],
            distances=torch.zeros((2, 2)),
            responder_cell_type=1,
            is_dividing=(second_cell_type < 2),
        )

    samples = [second_cell_type_sample(n) for n in range(5)]
    return CellMatesDataset(samples)


def generate_dataset_for_n_cells_test() -> CellMatesDataset:
    """
    Generates a dataset with 10 tissues, composed of 1-10 identical cells in the middle.
    The number of cells determines division.
    """

    def identical_cell_sample(n_cells: int) -> Sample:
        return Sample(
            cell_types=np.repeat(1, n_cells),
            distances=torch.zeros((n_cells, n_cells)),
            responder_cell_type=1,
            is_dividing=(n_cells % 2 == 0),
        )

    samples = [identical_cell_sample(n) for n in range(10)]
    return CellMatesDataset(samples)
