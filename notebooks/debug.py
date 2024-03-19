import torch
from cellmates.data.stubs import repeated_cell_sample
from cellmates.data import collate_fn
from cellmates.model.transformer import CellMatesTransformer
from fire import Fire


def run(b: int = 1):

    tr = CellMatesTransformer(D=512, K=int(512 / 16), num_encoder_layers=0)

    b1 = collate_fn([repeated_cell_sample(n) for n in [2, 10]])
    b2 = collate_fn([repeated_cell_sample(n) for n in [2, 5]])

    if b == 1:
        o1 = tr(
            cell_types_BL=b1["cell_types_BL"],
            distances_BLL=b1["distances_BLL"],
            padding_mask_BL=b1["padding_mask_BL"],
        )
    else:
        o2 = tr(
            cell_types_BL=b2["cell_types_BL"],
            distances_BLL=b2["distances_BLL"],
            padding_mask_BL=b2["padding_mask_BL"],
        )

    # torch.allclose(o1, o2)


if __name__ == "__main__":
    Fire(run)
