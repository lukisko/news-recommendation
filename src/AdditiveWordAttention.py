import torch
import torch.nn as nn
import torch.nn.functional as F

class AdditiveWordAttention(nn.Module):
    def __init__(self, hidden_size, attention_dim):
        """
        Initializes 3rd layer with the Additive Word Attention.

        Args:
            hidden_size (int): The size of the hidden embeddings (e.g., 768 for xlm-roberta-base).
            attention_dim (int): The dimensionality of the attention space.
        """
        super(AdditiveWordAttention, self).__init__()

        # Projection layer Vw: projects hidden_size to attention_dim
        self.Vw = nn.Linear(hidden_size, attention_dim, bias=True)

        # Query vector qw: projects attention_dim to a scalar score
        self.qw = nn.Linear(attention_dim, 1, bias=False)

        # Activation function
        self.tanh = nn.Tanh()

    def forward(self, h, mask=None):
        """
        Forward pass for the additive word attention layer.

        Args:
            h (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
            mask (torch.Tensor, optional): Mask tensor of shape (batch_size, seq_length),
                                          where elements with value `True` indicate valid tokens.

        Returns:
            torch.Tensor: News representations with shape (batch_size, hidden_size).
            torch.Tensor: Attention weights with shape (batch_size, seq_length).
        """
        # Apply linear projection and activation
        u = self.tanh(self.Vw(h))  # Shape: (batch_size, seq_length, attention_dim)

        # Compute attention scores
        a = self.qw(u).squeeze(-1)  # Shape: (batch_size, seq_length)

        # Apply mask: set scores of padded tokens to -inf
        if mask is not None:
            a = a.masked_fill(~mask, float('-inf'))

        # Compute attention weights
        alpha = F.softmax(a, dim=1)  # Shape: (batch_size, seq_length)

        # Compute the weighted sum of word embeddings
        r = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # Shape: (batch_size, hidden_size)

        return r, alpha
