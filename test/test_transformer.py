from cellmates.model.transformer import bucketize_distances
import pytest
import torch


@pytest.mark.parametrize(
    "distance, bin_idx",
    [
        (-2024, 0),
        (0, 0),
        (9, 0),
        (10, 0),
        (11, 1),
        (20, 1),
        (21, 2),
        (30, 2),
        (140, 13),
        (141, 14),
        (2024, 14),
    ],
)
def test_distance_discretization(distance, bin_idx):
    """"""
    distance_tensor = torch.tensor([distance])
    assert bucketize_distances(distance_tensor)[0] == bin_idx
