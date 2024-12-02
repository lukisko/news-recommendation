import torch.nn as nn
from AdditiveWordAttention import AdditiveWordAttention
from WordLevelMultiHeadSelfAttention import WordLevelMultiHeadSelfAttention
from XLMRobertaWordEmbedder import XLMRobertaWordEmbedder


class NewsEncoderModel(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, attention_dim=128, dropout=0.1):
        """
        Initializes the News Encoder Model with word embeddings, multi-head self-attention, and additive word attention.

        Args:
            hidden_size (int): The size of the hidden embeddings (e.g., 768 for xlm-roberta-base).
            num_heads (int): The number of attention heads.
            attention_dim (int): The dimensionality of the attention space for additive attention.
            dropout (float): Dropout probability.
        """
        super(NewsEncoderModel, self).__init__()

        # Initialize the first layer: XLM-RoBERTa Word Embedder
        self.word_embedder = XLMRobertaWordEmbedder()

        # Initialize the second layer: Word-Level Multi-Head Self-Attention
        self.self_attention = WordLevelMultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # Initialize the third layer: Additive Word Attention
        self.additive_attention = AdditiveWordAttention(
            hidden_size=hidden_size,
            attention_dim=attention_dim
        )

    def forward(self, titles):
        """
        Generates enhanced word embeddings and final news representations using XLM-RoBERTa, multi-head self-attention,
        and additive word attention.

        Args:
            titles (List[str]): A list of input news titles.

        Returns:
            torch.Tensor: Enhanced embeddings from self-attention with shape (batch_size, seq_length, hidden_size).
            torch.Tensor: Self-attention weights with shape (batch_size, num_heads, seq_length, seq_length).
            torch.Tensor: Final news representations with shape (batch_size, hidden_size).
            torch.Tensor: Additive attention weights with shape (batch_size, seq_length).
        """
        # Obtain word embeddings and attention masks from the first layer
        token_embeddings, attention_mask = self.word_embedder(titles)  # (batch_size, seq_length, hidden_size), (batch_size, seq_length)

        # Prepare the attention mask for the self-attention layer
        # nn.MultiheadAttention expects 'key_padding_mask' where True indicates padding tokens
        # The 'attention_mask' from the tokenizer has 1 for valid tokens and 0 for padding
        # Therefore, we invert it to get True for padding
        key_padding_mask = ~attention_mask.bool()  # Shape: (batch_size, seq_length)

        # Apply the multi-head self-attention layer
        enhanced_embeddings, self_attn_weights = self.self_attention(token_embeddings, attention_mask=key_padding_mask)
        # enhanced_embeddings: (batch_size, seq_length, hidden_size)
        # self_attn_weights: (batch_size, num_heads, seq_length, seq_length)

        # Apply the additive word attention layer
        # Prepare mask where True indicates valid tokens for additive attention
        additive_mask = attention_mask.bool()  # Shape: (batch_size, seq_length)

        final_representations, additive_attn_weights = self.additive_attention(enhanced_embeddings, mask=additive_mask)
        # final_representations: (batch_size, hidden_size)
        # additive_attn_weights: (batch_size, seq_length)

        return enhanced_embeddings, self_attn_weights, final_representations, additive_attn_weights
