import torch
import torch.nn as nn


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(
            torch.Tensor(max_relative_position * 2 + 1, num_units)
        )
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim  # D
        self.n_heads = n_heads  # H
        self.head_dim = hid_dim // n_heads  # K = D / H
        self.max_relative_position = 2

        self.relative_position_k = RelativePosition(
            self.head_dim, self.max_relative_position
        )
        self.relative_position_v = RelativePosition(
            self.head_dim, self.max_relative_position
        )

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query_BLD, key_BLD, value_BLD, mask=None):
        """
        query = [batch size, query len, hid dim]
        key = [batch size, key len, hid dim]
        value = [batch size, value len, hid dim]
        """
        batch_size = query_BLD.shape[0]
        len_k = key_BLD.shape[1]
        len_q = query_BLD.shape[1]
        len_v = value_BLD.shape[1]

        query_BLD = self.fc_q(query_BLD)
        key_BLD = self.fc_k(key_BLD)
        value_BLD = self.fc_v(value_BLD)

        r_q1_BLHK = query_BLD.view(batch_size, -1, self.n_heads, self.head_dim)
        r_q1_BHLK = r_q1_BLHK.permute(0, 2, 1, 3)

        r_k1_BLHK = key_BLD.view(batch_size, -1, self.n_heads, self.head_dim)
        r_k1_BHLK = r_k1_BLHK.permute(0, 2, 1, 3)
        r_k1_BHKL = r_k1_BHLK.permute(0, 1, 3, 2)
        attn1_BHLL = torch.matmul(r_q1_BHLK, r_k1_BHKL)

        r_q2_LBD = query_BLD.permute(1, 0, 2).contiguous()
        r_q2_L_BH_K = r_q2_LBD.view(len_q, batch_size * self.n_heads, self.head_dim)

        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)

        # combine parts and scale:
        attn_BHLL = (attn1_BHLL + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(
            0, 2, 1, 3
        )
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = (
            attn.permute(2, 0, 1, 3)
            .contiguous()
            .view(len_q, batch_size * self.n_heads, len_k)
        )
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = (
            weight2.transpose(0, 1)
            .contiguous()
            .view(batch_size, self.n_heads, len_q, self.head_dim)
        )

        x = weight1 + weight2

        # x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        # x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        # x = [batch size, query len, hid dim]

        return x


"""
Course:
"""


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
