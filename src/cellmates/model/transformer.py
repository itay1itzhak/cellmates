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

        # Encoder layers contain spatial multi-head attention:
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


class SpatialAttentionHead(nn.Module):
    """
    This class represents an attention head for transformer models.
    """

    def __init__(self, d_input: int, n_hidden: int):
        """
        Initializes the AttentionHead.

        Args:
            d_input: the dimension of the input
            n_hidden: the dimension of the keys, queries, and values
        """
        super().__init__()
        self.W_K = nn.Linear(d_input, n_hidden)
        self.W_Q = nn.Linear(d_input, n_hidden)
        self.W_V = nn.Linear(d_input, n_hidden)
        self.n_hidden = n_hidden

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the attention head.

        Args:
            x (torch.Tensor): The input tensor. Shape: (batch_size, seq_length, d_input)
            attn_mask (Optional[torch.Tensor]): The causal mask tensor. If provided, it acts as an attention mask
            that determines which tokens in the sequence should be attended to. It's a 3D tensor where the value at
            position [b, i, j] is 1 if the token at position i in batch b should attend to the token at position j,
            and 0 otherwise. If not provided (None), ignore it.
            Shape: (batch_size, seq_length, seq_length)

        Returns:
            attn_output (torch.Tensor): The output tensor after attention. Shape: (batch_size, seq_length, n_hidden)
            attn_score (torch.Tensor): The attention score tensor. Shape: (batch_size, seq_length, seq_length)
        """
        attn_output, attn_score = None, None

        # ======= Your Code Starts Here ========

        batch_size, n_words, embedding_dim = x.shape

        # compute Q,K,V matrices:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        assert Q.shape == K.shape == V.shape == (batch_size, n_words, self.n_hidden)

        # compute scores:
        scores = Q @ K.transpose(1, 2) / np.sqrt(self.n_hidden)

        # assign a score of negative infinity to non-attended tokens:
        if attn_mask is not None:
            scores = scores.masked_fill_(attn_mask == 0, -float("inf"))

        attn_score = nn.functional.softmax(scores, dim=-1)
        assert attn_score.shape == (batch_size, n_words, n_words)

        attn_output = attn_score @ V

        # ======= Your Code Ends Here ========

        return attn_output, attn_scor


class SpatialMultiheadAttention(nn.Module):
    def __init__(self, d_input: int, n_hidden: int, num_heads: int):
        """
        Initializes the MultiheadAttention.

        Args:
            d_input (int): The dimension of the input.
            n_hidden: the hidden dimenstion for the attention layer
            num_heads (int): The number of attention heads.
        Attributes:
            attention_heads (nn.ModuleList): A list of attention heads.
            W_proj (nn.Linear): A linear layer for projecting the concatenated outputs of the attention heads back
            to the original dimension.
        """

        super().__init__()

        # ======= Your Code Starts Here ========
        self.attention_heads = nn.ModuleList(
            [SpatialAttentionHead(d_input, n_hidden) for i in range(num_heads)]
        )
        self.W_proj = nn.Linear(num_heads * n_hidden, d_input)

        # for tests later:
        self.d_input = d_input
        self.n_hidden = n_hidden
        self.num_heads = num_heads

        # ======= Your Code Ends Here ========

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Executes the forward pass of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): The input tensor. It has a shape of (batch_size, seq_length, d_input).
            attn_mask (Optional[torch.Tensor]): The attention mask tensor. If provided, it serves as an attention guide
            that specifies which tokens in the sequence should be attended to. It's a 3D tensor where the value at
            position [b, i, j] is 1 if the token at position i in batch b should attend to the token at position j,
            and 0 otherwise. If not provided (None), ignore it.
            Shape: (batch_size, seq_length, seq_length)

        Returns:
            attn_output (torch.Tensor): The output tensor after applying multi-head attention. It has a shape of
            (batch_size, seq_length, d_input).

        This method computes the multi-head attention by looping through each attention head, collecting the outputs,
        concatenating them together along the hidden dimension, and then projecting them back into the output dimension
        (d_input). It returns both the final attention outputs as well as the attn_scores from each head.
        """
        attn_output, attn_scores = None, None

        # ======= Your Code Starts Here ========
        batch_size, n_words, embedding_dim = x.shape  # for tests

        # run all heads:
        attn_output_per_head = []
        attn_scores = []
        for head in self.attention_heads:
            o, s = head(x, attn_mask)
            attn_output_per_head.append(o)
            attn_scores.append(s)

        stacked_attn_output = torch.concat(attn_output_per_head, dim=-1)
        assert stacked_attn_output.shape == (
            batch_size,
            n_words,
            self.num_heads * self.n_hidden,
        )

        attn_scores = torch.stack(attn_scores).transpose(0, 1)
        assert attn_scores.shape == (batch_size, self.num_heads, n_words, n_words)

        # project:
        attn_output = self.W_proj(stacked_attn_output)
        assert attn_output.shape == (batch_size, n_words, self.d_input)

        # ======= Your Code Ends Here ========
        return attn_output, attn_scores
