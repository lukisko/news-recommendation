import torch
import torch.nn as nn
from AdditiveWordAttention import AdditiveWordAttention
from PytorchMultiHeadSelfAttHead import PytorchMultiHeadSelfAttHead
from ManualMultiHeadSelfAttHead import ManualMultiHeadSelfAttHead
from XLMRobertaWordEmbedder import XLMRobertaWordEmbedder


class NewsEncoder(nn.Module):
    def __init__(self, hidden_size=768, num_heads=16, attention_dim=128, dropout=0.1):
        """
        Initializes the News Encoder Model with word embeddings, multi-head self-attention, and additive word attention.

        Args:
            hidden_size (int): The size of the hidden embeddings (e.g., 768 for xlm-roberta-base).
            num_heads (int): The number of attention heads.
            attention_dim (int): The dimensionality of the attention space for additive attention.
            dropout (float): Dropout probability.
        """
        super(NewsEncoder, self).__init__()
        assert hidden_size % num_heads == 0, "Embeding must be divisible by heads"


        self.hidden_size = hidden_size

        # Initialize the first layer: XLM-RoBERTa Word Embedder
        self.word_embedder = XLMRobertaWordEmbedder()

        # Initialize the second layer: Word-Level Multi-Head Self-Attention
        # self.self_attention = PytorchMultiHeadSelfAttHead(
        #     hidden_size=hidden_size,
        #     num_heads=num_heads,
        #     dropout=dropout
        # )

        self.self_attention = ManualMultiHeadSelfAttHead(
            embedding_dimension=hidden_size,
            head_count=num_heads
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

        input_shape = titles.shape
        flatten_titles = titles.flatten()
        titles_list = flatten_titles.tolist()

        # Obtain word embeddings and attention masks from the first layer
        token_embeddings, attention_mask = self.word_embedder(titles_list)  # (batch_size, seq_length, hidden_size), (batch_size, seq_length)
        output = token_embeddings.reshape(input_shape + (30, self.hidden_size))

        # Prepare the attention mask for the self-attention layer
        # nn.MultiheadAttention expects 'key_padding_mask' where True indicates padding tokens
        # The 'attention_mask' from the tokenizer has 1 for valid tokens and 0 for padding
        # Therefore, we invert it to get True for padding
        key_padding_mask = ~attention_mask.bool()  # Shape: (batch_size, seq_length)

        # Apply the multi-head self-attention layer

        print(output.shape)

        # For PytorchMultiHeadSelfAttHead
        enhanced_embeddings, self_attn_weights = self.self_attention(output, attention_mask=key_padding_mask)

        self_attn_weights = torch.ones(6, 30, 30)

        # For ManualMultiHeadSelfAttHead
        enhanced_embeddings = self.self_attention(output)

        # Apply the additive word attention layer
        # Prepare mask where True indicates valid tokens for additive attention
        additive_mask = attention_mask.bool()  # Shape: (batch_size, seq_length)

        print(enhanced_embeddings.shape)
        final_representations, additive_attn_weights = self.additive_attention(enhanced_embeddings, mask=additive_mask)
        # final_representations: (batch_size, hidden_size)
        # additive_attn_weights: (batch_size, seq_length)

        return enhanced_embeddings, self_attn_weights, final_representations, additive_attn_weights
