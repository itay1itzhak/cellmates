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
            is_dividing=second_cell_distance < 40,
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
            is_dividing=(second_cell_type < 3),
        )

    samples = [second_cell_type_sample(n) for n in range(1,6)]
    return CellMatesDataset(samples)


def repeated_cell_sample(n_cells: int, threshold: int = 5) -> Sample:
    return Sample(
        cell_types=np.repeat(1, n_cells),
        distances=torch.zeros((n_cells, n_cells)),
        responder_cell_type=1,
        # is_dividing=(n_cells > threshold),
        is_dividing=(n_cells > threshold),
    )


def generate_dataset_for_n_cells_test(n=10, step=1) -> CellMatesDataset:
    """
    Generates a dataset with 10 tissues, composed of 1-10 identical cells in the middle.
    The number of cells determines division.
    """
    # samples = [
    #     repeated_cell_sample(i, threshold=(n - 1) * step / 2)
    #     for i in range(1, (n + 1) * step, step)
    # ]

    samples = [
        repeated_cell_sample(i, threshold=15)
        for i in [5, 5, 5, 5, 5, 30, 30, 30, 30, 30, 30, 30]
    ]
    return CellMatesDataset(samples)
