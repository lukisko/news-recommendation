import torch.nn as nn


class PytorchMultiHeadSelfAttHead(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        """
        Initializes the 2nd layer with the Word-Level Multi-Head Self-Attention.

        Args:
            hidden_size (int): The size of the hidden embeddings (e.g., 768 for xlm-roberta-base).
            num_heads (int): The number of attention heads.
            dropout (float): Dropout probability for attention weights.
        """
        super(PytorchMultiHeadSelfAttHead, self).__init__()

        # Multi-head attention module
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Ensures input/output tensors are (batch, seq, feature)
        )

    def forward(self, x, attention_mask=None):
        """
        Forward pass for the multi-head self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, hidden_size).
            attention_mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_length),
                                                     where elements with value `True` are masked.

        Returns:
            torch.Tensor: Output tensor after self-attention and residual connection,
                          shape (batch_size, seq_length, hidden_size).
            torch.Tensor: Attention weights of shape (batch_size, num_heads, seq_length, seq_length).
        """
        # Apply multi-head self-attention
        # Note: nn.MultiheadAttention expects inputs of shape (batch, seq, feature) with batch_first=True
        print(x.shape)
        attn_output, attn_weights = self.multihead_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=attention_mask  # Masks padded tokens if provided
        )

        # Apparently not used in the paper.
        # TODO:
        #   This be an idea to improve the model, maybe bring back with it the normalization.
        # Add residual connections
        # x = x + attn_output

        return x, attn_weights
