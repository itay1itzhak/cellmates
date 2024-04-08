import torch

from cellmates.data.sample import Sample
from cellmates.data.breast import BreastCancerTissueDataset
from cellmates.utils import MAX_EFFECTIVE_DISTANCE, CELL_TYPE_STR_TO_IDX

from tdm.tissue import StubTissue
from tdm.utils import microns
from tdm.cell_types import FIBROBLAST, MACROPHAGE


def test_distances_reorder():
    st = StubTissue(
        cell_types=[FIBROBLAST],
        cell_type_xy_tuples=[
            (FIBROBLAST, microns(500), microns(500), 0), # located at 500,500, not dividing
            (FIBROBLAST, microns(500), microns(500), 0),
            (FIBROBLAST, microns(500), microns(600), 1)
        ],
        tissue_dimensions=(microns(1000), microns(1000))
    )

    ds = BreastCancerTissueDataset(
        tissue=st, 
        effective_distance=MAX_EFFECTIVE_DISTANCE, 
        responder_cell_type=FIBROBLAST
    )

    sample_01 = Sample(
        cell_types=torch.tensor([1,1,1]),
        is_dividing=False,
        responder_cell_type=CELL_TYPE_STR_TO_IDX[FIBROBLAST],
        distances=torch.tensor(
            [[0,0,100],
             [0,0,100],
             [100,100,0]]
        )
    )

    assert ds[0] == sample_01
    assert ds[1] == sample_01

    sample_2 = Sample(
        cell_types=torch.tensor([1,1,1]),
        is_dividing=True,
        responder_cell_type=CELL_TYPE_STR_TO_IDX[FIBROBLAST],
        distances=torch.tensor(
            [[0,100,100],
             [100,0,0],
             [100,0,0]]
        )
    )

    assert ds[2] == sample_2


def test_cell_types_reorder():
    st = StubTissue(
        cell_types=[FIBROBLAST, MACROPHAGE],
        cell_type_xy_tuples=[
            (FIBROBLAST, microns(500), microns(500), 0), # located at 500,500, not dividing
            (FIBROBLAST, microns(500), microns(500), 0),
            (MACROPHAGE, microns(500), microns(550.1), 0),
            (FIBROBLAST, microns(500), microns(600), 1)
        ],
        tissue_dimensions=(microns(1000), microns(1000))
    )
    ds = BreastCancerTissueDataset(
        tissue=st, 
        effective_distance=MAX_EFFECTIVE_DISTANCE, 
        responder_cell_type=FIBROBLAST
    )

    f = CELL_TYPE_STR_TO_IDX[FIBROBLAST]
    m = CELL_TYPE_STR_TO_IDX[MACROPHAGE]

    assert torch.allclose(ds[0].cell_types, torch.tensor([f,f,m,f]))
    assert torch.allclose(ds[1].cell_types, torch.tensor([f,f,m,f]))
    assert torch.allclose(ds[2].cell_types, torch.tensor([f,f,f,m]))