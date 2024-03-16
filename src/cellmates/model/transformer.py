import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Dropout,
    LayerNorm,
    Embedding,
    ModuleDict,
    ModuleList,
)
from typing import Callable
from math import sqrt


# We discretize distances in jumps of 10 microns.
# bins: <10, [10,20), ... [130,140), [140,infty)
# <10 corresponds with direct contact.
# we assume communication in distances >140 should be equally inneffective
DISTANCE_BINS = torch.arange(10, 150, 10)
N_DISTANCES = len(DISTANCE_BINS) + 1

RESPONDER_CELL_IDX = 0


def bucketize_distances(distances: Tensor):
    return torch.bucketize(distances, DISTANCE_BINS.to(distances.device), right=False)


class CellMatesTransformer(nn.Module):
    """
    Transformer Dimension Notation:

    B: batch size
    L: sequence length (number of cells)
    D: model dimension (cell-type embedding dimension)
    V: vocabulary size (number of cell types)
    F: feedfoward subnetwork hidden size
    H: number of attention heads in a layer
    K: size of each attention key or value (also dimension of distance embeddings)
    M: mlp hidden size
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        D: int,
        H: int = 16,
        K: int = 512,
        F: int = 2048,
        M: int = 512,
        n_cell_types: int = 6,
        num_encoder_layers: int = 8,
        dropout_p: float = 0.1,
        activation: str | Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """_summary_"""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # cells are embedded according to type:
        self.cell_type_embedding = Embedding(
            num_embeddings=n_cell_types, embedding_dim=D
        )

        # relative distance embeddings (dim=K) are added to key and value vectors in
        # the multi-head attention layer
        # we include three relative distance embeddings - qk, qr, kr
        # q = query, k = key, r = responder
        self.distance_embeddings = ModuleDict(
            {
                k: Embedding(num_embeddings=N_DISTANCES, embedding_dim=K)
                for k in ["Kqk", "Kqr", "Kkr", "Vqk", "Vqr", "Vkr"]
            }
        )

        # Encoder layers perform spatial multi-head attention:
        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                CellMatesEncoderLayer(
                    D=D,
                    H=H,
                    K=K,
                    F=F,
                    dropout_p=dropout_p,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    norm_first=norm_first,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )
        self.encoder_layers = ModuleList(encoder_layers)

        # MLP computes a single probability from pooled cell-representations:
        self.mlp_linear1 = Linear(D, M, bias=bias, **factory_kwargs)
        self.mlp_dropout = Dropout(dropout_p)
        self.mlp_linear2 = Linear(M, 1, bias=bias, **factory_kwargs)

    def forward(
        self, cell_types_BL: Tensor, distances_BLL: Tensor, padding_mask_BL: Tensor
    ) -> Tensor:

        hidden_BLD = self.cell_type_embedding(cell_types_BL)

        distance_idxs_BLL = bucketize_distances(distances_BLL)

        # Apply encoder layers with distance embeddings
        for layer in self.encoder_layers:
            hidden_BLD = layer(
                hidden_BLD, distance_idxs_BLL, padding_mask_BL, self.distance_embeddings
            )

        # Pooling - sum without padding vectors
        output_BD = torch.einsum("BLD,BL->BD", hidden_BLD, padding_mask_BL)

        # Apply MLP to the pooled cell-representations:
        output_BM = self.mlp_dropout(F.relu(self.mlp_linear1(output_BD)))
        output_B1 = self.mlp_linear2(output_BM)

        return output_B1


class CellMatesEncoderLayer(nn.Module):
    """
    Transformer Dimension Notation:

    B: batch size
    L: sequence length (number of cells)
    D: model dimension (cell-type embedding dimension)
    V: vocabulary size (number of cell types)
    F: feedfoward subnetwork hidden size
    H: number of attention heads in a layer
    K: size of each attention key or value (also dimension of distance embeddings)
    M: mlp hidden size
    """

    def __init__(
        self,
        D: int,  # D
        H: int,  # H
        K: int = 512,  # K
        F: int = 2048,  # F
        dropout_p: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        # batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """_summary_"""
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.attn = SpatialMultiHeadAttention(D, H, K, device)

        # Implementation of Feedforward model
        self.linear1 = Linear(D, F, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout_p)
        self.linear2 = Linear(F, D, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(D, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.norm2 = LayerNorm(D, eps=layer_norm_eps, bias=bias, **factory_kwargs)  # type: ignore
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)

        self.activation = activation

    def forward(
        self,
        x_BLD: Tensor,
        distance_idxs_BLL: Tensor,
        padding_mask_BL: Tensor,
        distance_embeddings: ModuleDict,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        if self.norm_first:
            x_BLD = x_BLD + self._sa_block(
                self.norm1(x_BLD),
                distance_idxs_BLL=distance_idxs_BLL,
                distance_embeddings=distance_embeddings,
                padding_mask_BL=padding_mask_BL,
            )
            x_BLD = x_BLD + self._ff_block(self.norm2(x_BLD))
        else:
            x_BLD = self.norm1(
                x_BLD
                + self._sa_block(
                    x_BLD,
                    distance_idxs_BLL=distance_idxs_BLL,
                    distance_embeddings=distance_embeddings,
                    padding_mask_BL=padding_mask_BL,
                )
            )
            x_BLD = self.norm2(x_BLD + self._ff_block(x_BLD))

        return x_BLD

    # self-attention block
    def _sa_block(
        self,
        x_BLD: Tensor,
        distance_idxs_BLL: Tensor,
        distance_embeddings: ModuleDict,
        padding_mask_BL: Tensor | None = None,
    ) -> Tensor:
        x = self.attn(
            x_BLD,
            distance_idxs_BLL,
            distance_embeddings,
            padding_mask_BL=padding_mask_BL,  # for differenet length batches
        )
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class SpatialMultiHeadAttention(nn.Module):
    def __init__(
        self,
        D: int,  # D
        H: int,  # H
        K: int,  # K
        device: str | None = None,
    ) -> None:
        super().__init__()

        # save shapes:
        self.D = D
        self.H = H
        self.K = K

        # init multi-head weight matrices:
        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(D, D)
        self.Wv = nn.Linear(D, D)
        self.Wo = nn.Linear(D, D)

        # init scaler and move to device:
        self.sqrt_K = torch.FloatTensor([sqrt(K)]).to(device)

    def forward(
        self,
        x_BLD: Tensor,
        distance_idxs_BLL: Tensor,
        distance_embeddings: ModuleDict,
        padding_mask_BL: Tensor | None = None,
    ):

        # set dims:
        B, L, D = x_BLD.shape
        H, K = self.H, self.K

        # apply linear layers:
        Q_BLD = self.Wq(x_BLD)
        K_BLD = self.Wk(x_BLD)
        V_BLD = self.Wv(x_BLD)

        # reshape for separating matmuls per-head:
        Q_BLHK = Q_BLD.view(B, L, H, K)
        K_BLHK = K_BLD.view(B, L, H, K)
        V_BLHK = V_BLD.view(B, L, H, K)

        """
        Attention score components:
        """
        # 1 - vanilla:
        E_BHLL = torch.einsum("BLHK,BXHK->BHLX", Q_BLHK, K_BLHK)
        E_BHLL = E_BHLL

        # 2 - qk distances:
        Kqk_BLLK = distance_embeddings["Kqk"](distance_idxs_BLL)
        Eqk_BHLL = torch.einsum("BLHK,BLXK -> BHLX", Q_BLHK, Kqk_BLLK)

        # 3 - qr distances
        dist_to_r_idxs_BL = distance_idxs_BLL[:, RESPONDER_CELL_IDX]
        Kqr_BLK = distance_embeddings["Kqr"](dist_to_r_idxs_BL)
        Eqr_BHL = torch.einsum("BLHK,BLK -> BHL", Q_BLHK, Kqr_BLK)
        # identical score for all j:
        Eqr_BHLL = Eqr_BHL.unsqueeze(-1).expand((B, H, L, L))

        # 4 - kr distances
        Kkr_BLK = distance_embeddings["Kkr"](dist_to_r_idxs_BL)
        Ekr_BHLL = torch.einsum("BLHK,BXK -> BHLX", Q_BLHK, Kkr_BLK)

        # sum:
        E_BHLL = E_BHLL + Eqk_BHLL + Eqr_BHLL + Ekr_BHLL

        # -inf score for padding vectors:
        padding_mask_BHLL = padding_mask_BL.repeat(1, H, 1, L).reshape((B, H, L, L))
        E_BHLL.masked_fill_(padding_mask_BHLL == 0, -float("inf"))

        E_BHLL = torch.softmax(E_BHLL, dim=-1)

        """
        Value vector components:
        """
        # 1 - vanilla:
        Z_BLHK = torch.einsum("BHLX,BXHK->BLHK", E_BHLL, V_BLHK)

        # 2 - qk:
        Vqk_BLLK = distance_embeddings["Vqk"](distance_idxs_BLL)
        Zqk_BLHK = torch.einsum("BHLX,BLXK->BLHK", E_BHLL, Vqk_BLLK)

        # 3 - qr:
        Zqr_BLK = distance_embeddings["Vqr"](
            dist_to_r_idxs_BL
        )  # _Z_qr is not a bug! Weights sum to one
        Zqr_BHLK = Zqr_BLK.unsqueeze(1).expand((B, H, L, K))
        Zqr_BLHK = Zqr_BHLK.permute(0, 2, 1, 3)

        # 4 - kr:
        Vkr_BLK = distance_embeddings["Vkr"](dist_to_r_idxs_BL)
        Zkr_BLHK = torch.einsum("BHLX,BXK->BLHK", E_BHLL, Vkr_BLK)

        # sum value components:
        Z_BLHK = Z_BLHK + Zqk_BLHK + Zqr_BLHK + Zkr_BLHK

        # concat all head computations:
        Z_BLD = Z_BLHK.reshape(B, L, D)

        output_BLD = self.Wo(Z_BLD)

        return output_BLD
