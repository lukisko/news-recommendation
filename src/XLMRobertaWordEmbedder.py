import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel


class XLMRobertaWordEmbedder(nn.Module):
    def __init__(self):
        """
        Initializes the tokenizer and model from the specified pretrained XLM-RoBERTa model.
        """
        super(XLMRobertaWordEmbedder, self).__init__()

        # Initialize the tokenizer
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        # Initialize the model
        self.model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        # Set the model to evaluation mode to deactivate dropout layers
        self.model.eval()

    def forward(self, titles):
        """
        Generates word embeddings for the provided input list of titles.

        Args:
            titles (List[str]): A list of input titles.

        Returns:
            torch.Tensor: Tensor containing word embeddings with shape (batch_size, seq_length, hidden_size).
        """
        # Tokenize the input titles
        encoded_input = self.tokenizer(
            titles,                      # List of titles to encode
            padding='max_length',        # Pad all sequences to the max_length
            truncation=True,             # Truncate sentences longer than max_length
            max_length=30,               # Define a fixed max_length
            return_tensors='pt',         # Return PyTorch tensors
            return_attention_mask=True,  # Return attention masks
            return_token_type_ids=False  # XLM-RoBERTa doesn't use token type IDs
        )

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():  # Disable gradient computation
            outputs = self.model(**encoded_input)

        # Extract the last hidden states (token embeddings)
        token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        attention_mask = encoded_input['attention_mask']  # Shape: (batch_size, seq_length)

        return token_embeddings, attention_mask
