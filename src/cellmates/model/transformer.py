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
N_DISTANCES = len(DISTANCE_BINS)


def bucketize_distances(distances: Tensor):
    return torch.bucketize(distances, DISTANCE_BINS, right=False)


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
                for k in ["K_qk", "K_qr", "K_kr", "V_qk", "V_qr", "V_kr"]
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
        self,
        cell_types_BL: Tensor,
        distances_BLL: Tensor,
    ) -> Tensor:

        hidden_BLD = self.cell_type_embedding(cell_types_BL)

        distance_idxs_BLL = bucketize_distances(distances_BLL)

        # Apply encoder layers with distance embeddings
        for layer in self.encoder_layers:
            hidden_BLD = layer(hidden_BLD, distance_idxs_BLL, self.distance_embeddings)

        # Apply MLP to the pooled cell-representations:
        output_BD = hidden_BLD.sum(dim=1)
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
        activation: str | Callable[[Tensor], Tensor] = F.gelu,
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

        self.attn = SpatialMultiHeadAttention()

        # Implementation of Feedforward model
        self.linear1 = Linear(D, F, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout_p)
        self.linear2 = Linear(F, D, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(D, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(D, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout_p)
        self.dropout2 = Dropout(dropout_p)

        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
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
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
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
        device: str,
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

    def forward(self, x_BLD):

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

        # attention scores:
        E_BHLL = torch.einsum("BLHK,BXHK->BHLX", Q_BLHK, K_BLHK)
        E_BHLL = E_BHLL / self.sqrt_K
        E_BHLL = torch.softmax(E_BHLL, dim=-1)

        # BHLL,BLHK->BHLK
        Z_BLHK = torch.einsum("BHLX,BXHK->BLHK", E_BHLL, V_BLHK)

        # concat all head computations:
        Z_BLD = Z_BLHK.reshape(B, L, D)

        output_BLD = self.Wo(Z_BLD)

        return output_BLD
