from typing import Literal, overload
import numpy as np
from scipy.spatial.distance import pdist, squareform

from cellmates.data.dataset import CellMatesDataset
from cellmates.data.sample import Sample
from cellmates.utils import CELL_TYPE_STR_TO_IDX
from cellmates.cache import persistent_cache

from tdm.raw.raza_breast_mibi.utils import all_image_numbers
from tdm.tissue import Tissue, RazaBreast

import torch
from torch.utils.data import ConcatDataset, Dataset
from cellmates.utils import MAX_EFFECTIVE_DISTANCE

from copy import deepcopy


class BreastCancerTissueDataset(CellMatesDataset):
    def __init__(
        self,
        tissue: int | Tissue,
        effective_distance: int,
        responder_cell_type: str,
        mask_third_party_cells_distances: bool = False,
    ) -> None:
        self.responder_cell_type = responder_cell_type
        self.effective_distance = effective_distance
        self.tissue = self._init_tissue(tissue)
        self.valid_cell_idxs = (
            self._valid_cell_idxs()
        )  # use only cells whose complete neighborhood is visible
        self.distances = self._distances()
        self.mask_third_party_cells_distances = mask_third_party_cells_distances

    def __len__(self) -> int:
        return len(self.valid_cell_idxs)

    def __getitem__(self, item_idx: int) -> Sample:

        # fetch absolute index of cell in the original dataframe:
        cell_idx = self.valid_cell_idxs[item_idx]

        # fetch indexes of cells within effective_distance from the responder cell:
        nearby_cell_idxs = np.argwhere(
            self.distances[cell_idx] < self.effective_distance
        ).flatten()

        # sub-sample the complete distances matrix:
        distances_to_nearby_cells = self.distances[nearby_cell_idxs][
            :, nearby_cell_idxs
        ]

        # order of the responder cell among the cells in the neighborhood:
        responder_cell_idx = np.argwhere(nearby_cell_idxs == cell_idx).flatten()[0]

        # swap the row of the target cell to make it first:
        distances_to_nearby_cells[[0, responder_cell_idx]] = distances_to_nearby_cells[
            [responder_cell_idx, 0]
        ]

        # swap the column of the target cell to make it first:
        distances_to_nearby_cells[:, [0, responder_cell_idx]] = (
            distances_to_nearby_cells[:, [responder_cell_idx, 0]]
        )

        # fetch cell types and swap the responder cell's location again:
        df = self.tissue.cell_df()
        cell_types = df.loc[nearby_cell_idxs, "integer_cell_type"].values
        responder_cell_type = cell_types[responder_cell_idx]
        cell_types_without_responder = np.delete(cell_types, responder_cell_idx)
        cell_types = np.insert(cell_types_without_responder, 0, responder_cell_type)

        # mask third party cells distances for distance ablation study:
        if self.mask_third_party_cells_distances:
            distances_to_nearby_cells[1:, 1:] = (
                self.effective_distance
            )  # mask all distances except the responder cell

        return Sample(
            cell_types=torch.tensor(cell_types, dtype=int),
            distances=torch.tensor(distances_to_nearby_cells, dtype=int),
            responder_cell_type=responder_cell_type,
            is_dividing=df.loc[cell_idx, "division"],
        )

    def _distances(self) -> np.ndarray:
        """Pairwise distance matrix of distances in microns.

        Returns:
            np.ndarray: n x n distance matrix
        """
        xy_coords = self.tissue.cell_df()[["x", "y"]]
        return squareform(pdist(xy_coords)).astype(int)

    def _init_tissue(self, tissue: int | Tissue) -> Tissue:

        if np.isscalar(tissue):  # np.int64 doesn't pass isinstance(_,int)
            tissue = RazaBreast(tissue)

        tissue = deepcopy(tissue)

        df = tissue.cell_df()

        # we work with "integer microns", i.e 100 microns are just "100"
        df.loc[:, ["x", "y"]] = df.loc[:, ["x", "y"]] * 1e6
        df = df.astype({"x": int, "y": int}).reset_index(drop=True)

        # convert original dimensions to int:
        x_max, y_max = tissue._tissue_dimensions
        tissue._tissue_dimensions = int(x_max * 1e6), int(y_max * 1e6)

        # map string cell types to integers:
        df = df.assign(integer_cell_type=df.cell_type.map(CELL_TYPE_STR_TO_IDX))

        tissue._cell_df = df

        return tissue

    def _valid_cell_idxs(self) -> np.ndarray:
        """Indexes of cells that more than `effective_distance` away from tissue limits.

        Returns:
            np.ndarray: _description_
        """
        cell_df = self.tissue.cell_df()

        # each dataset fetches only cells of one kind:
        is_correct_type = cell_df.cell_type == self.responder_cell_type

        x_min, y_min = 0, 0
        x_max, y_max = self.tissue.tissue_dimensions()

        # compute x,y masks:
        xys = cell_df[["x", "y"]].to_numpy()
        ds_x, ds_y = xys[:, 0], xys[:, 1]
        x_is_in_bounds = (ds_x <= (x_max - self.effective_distance)) & (
            ds_x >= (x_min + self.effective_distance)
        )
        y_is_in_bounds = (ds_y <= (y_max - self.effective_distance)) & (
            ds_y >= (y_min + self.effective_distance)
        )

        # final mask:
        is_valid = x_is_in_bounds & y_is_in_bounds & is_correct_type

        return cell_df.index[is_valid].values


# overload get_datasets to account for different types of concatenated vs not:
@overload
def get_datasets(
    responder_cell_type: str,
    effective_distance: int,
    concatenated: Literal[False] = False,
) -> list[Dataset]: ...


@overload
def get_datasets(
    responder_cell_type: str,
    effective_distance: int,
    concatenated: Literal[True],
) -> Dataset: ...


def get_datasets(
    responder_cell_type: str,
    effective_distance: int,
    concatenated: Literal[True, False] = False,
    mask_third_party_cells_distances: bool = False,
) -> Dataset | list[Dataset]:
    """Wrapper for the cached function `_get_datasets`.

    Args:
        responder_cell_type (str): identifier of the type (see `CELL_TYPES_ARRAY` in `cellmates.utils`)
        effective_distance (int): number of microns we consider around the reponder cell.
        concatenated (Literal[True, False], optional): Defaults to False.
        mask_third_party_cells_distances (bool, optional): Defaults to False.

    Returns:
        Dataset | list[Dataset]: _description_
    """
    dss = _get_datasets(
        responder_cell_type=responder_cell_type,
        effective_distance=effective_distance,
        mask_third_party_cells_distances=mask_third_party_cells_distances,
    )

    if concatenated:
        return ConcatDataset(dss)
    else:
        return dss


@persistent_cache
def _get_datasets(
    responder_cell_type: str,
    effective_distance: int,
    mask_third_party_cells_distances: bool,
):
    return [
        BreastCancerTissueDataset(
            tissue=idx,
            effective_distance=effective_distance,
            responder_cell_type=responder_cell_type,
            mask_third_party_cells_distances=mask_third_party_cells_distances,
        )
        for idx in all_image_numbers()
    ]


def generate_one_tissue_dataset() -> CellMatesDataset:

    one_tissue_ds = BreastCancerTissueDataset(
        tissue=1, effective_distance=MAX_EFFECTIVE_DISTANCE, responder_cell_type="F"
    )

    samples = [s for s in one_tissue_ds]

    return CellMatesDataset(samples)
