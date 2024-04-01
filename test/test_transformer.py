from cellmates.model.transformer import bucketize_distances
import pytest
import torch
from cellmates.data.stubs import repeated_cell_sample
from cellmates.data import CellMatesDataset, collate_fn
from cellmates.model.transformer import CellMatesTransformer


@pytest.mark.parametrize(
    "distance, bin_idx",
    [
        (-2024, 15),
        (0, 0),
        (9, 0),
        (10, 1),
        (19, 1),
        (20, 2),
        (29, 2),
        (30, 3),
        (140, 14),
        (149, 14),
        (2024, 14),
    ],
)
def test_distance_discretization(distance, bin_idx):
    """"""
    distance_tensor = torch.tensor([distance])
    assert bucketize_distances(distance_tensor)[0] == bin_idx


def test_samples_are_independent_wrt_n_cells():

    tr = CellMatesTransformer(D=512, K=int(512 / 16), num_encoder_layers=1).eval()

    b1 = collate_fn([repeated_cell_sample(n) for n in [2, 10]])
    b2 = collate_fn([repeated_cell_sample(n) for n in [2, 5]])

    o1 = tr(
        cell_types_BL=b1["cell_types_BL"],
        distances_BLL=b1["distances_BLL"],
        padding_mask_BL=b1["padding_mask_BL"],
    )

    o2 = tr(
        cell_types_BL=b2["cell_types_BL"],
        distances_BLL=b2["distances_BLL"],
        padding_mask_BL=b2["padding_mask_BL"],
    )

    assert torch.allclose(o1[0], o2[0])
