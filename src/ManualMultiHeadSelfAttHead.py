import torch
import torch.nn as nn

class SelfAttHead(nn.Module):
    """
    A single self-attention head module.

    This module performs the self-attention mechanism on input embeddings. It computes
    queries and keys using a linear transformation, calculates attention scores, applies
    softmax to obtain attention weights, and finally computes the weighted sum of values.

    Attributes:
        lin_qk (nn.Linear): Linear layer to compute queries and keys from the input embeddings.
        softmax_dim1 (nn.Softmax): Softmax layer applied along the first dimension (dim=1).
        lin_vk (nn.Linear): Linear layer to compute the output values from the weighted sum.
    """

    def __init__(self, dim_emb: int, head_out: int):
        """
        Initializes the SelfAttHead module.

        Args:
            dim_emb (int): Dimension of the input embeddings.
            head_out (int): Dimension of the output for this attention head.
        """
        super().__init__()
        # Linear transformation for queries and keys (shared weights, no bias)
        self.lin_qk = nn.Linear(dim_emb, dim_emb, bias=False)

        # Softmax layer to normalize attention scores across the sequence
        self.softmax_dim1 = nn.Softmax(dim=1)  # Consider verifying if dim=1 is appropriate

        # Linear transformation for values to project to the desired head output dimension
        self.lin_vk = nn.Linear(in_features=dim_emb, out_features=head_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SelfAttHead module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, dim_emb).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, shape (batch_size, sequence_length, head_out).
        """
        # Compute queries and keys: Q_k^w e_j
        qe = self.lin_qk(x)  # Shape: (batch_size, sequence_length, dim_emb)

        # Compute attention scores: e_i^T Q_k^w e_j
        et_qt = x @ qe.transpose(-2, -1)  # Shape: (batch_size, sequence_length, sequence_length)

        # Apply softmax to obtain attention weights: a_{i,j}^k = softmax(e_i^T Q_k^w e_j)
        ak = self.softmax_dim1(et_qt)  # Shape: (batch_size, sequence_length, sequence_length)

        # Compute weighted sum of values: SUM_j a_{i,j}^k e_j
        weighted_sum = ak @ x  # Shape: (batch_size, sequence_length, dim_emb)

        # Project the weighted sum to the head's output dimension: V_k^w (weighted_sum)
        hk = self.lin_vk(weighted_sum)  # Shape: (batch_size, sequence_length, head_out)

        return hk

class ManualMultiHeadSelfAttHead(nn.Module):
    """
    Multi-Head Self-Attention module.

    This module aggregates multiple self-attention heads to capture information from different representation subspaces
    at different positions. The outputs from each head are concatenated to form the final output.

    Attributes:
        head_out (int): Output dimension for each individual attention head.
        selfAtt (nn.ModuleList): A list of SelfAttHead modules representing each attention head.
    """

    def __init__(self, embedding_dimension: int, head_count: int):
        """
        Initializes the MultiHeadSelfAttHead module.

        Args:
            embedding_dimension (int): Dimension of the input embeddings.
            head_count (int): Number of attention heads.
        """
        super().__init__()
        # Determine the output dimension per head
        self.head_out = embedding_dimension // head_count  # Ensure embedding_dimension is divisible by head_count

        # Create a list of SelfAttHead modules for each attention head
        self.selfAtt = nn.ModuleList([
            SelfAttHead(embedding_dimension, self.head_out) for _ in range(head_count)
        ])

    def forward(self, e_s: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiHeadSelfAttHead module.

        Args:
            e_s (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dimension).

        Returns:
            torch.Tensor: Concatenated output from all attention heads, shape (batch_size, sequence_length, embedding_dimension).
        """
        hk = []  # List to store outputs from each attention head

        # Iterate over each attention head and compute its output
        for head in self.selfAtt:
            att = head(e_s)  # Shape: (batch_size, sequence_length, head_out)
            hk.append(att)

        # Concatenate the outputs from all heads along the last dimension (embedding dimension)
        h = torch.cat(hk, dim=-1)  # Shape: (batch_size, sequence_length, embedding_dimension)

        return h
