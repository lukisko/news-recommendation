import torch
import numpy as np
import time
from NewsEncoder import NewsEncoder

# Initialize the model
encoder = NewsEncoder()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
encoder.eval()

# Example titles
titles_input = np.array(["Natascha var ikke den første", "Kun Star Wars tjente mere", "Luderne flytter på landet"])

# Generate embeddings and representations
with torch.no_grad():
    enhanced_embeddings, self_attn_weights, final_representations, additive_attn_weights = encoder(titles_input)

# Print output shapes
print("Enhanced Embedding Shape:", enhanced_embeddings.shape)  # Expected: (batch_size, seq_length, hidden_size)
print("Self-Attention Weights Shape:", self_attn_weights.shape)  # Expected: (batch_size, num_heads, seq_length, seq_length)
print("Final Representations Shape:", final_representations.shape)  # Expected: (batch_size, hidden_size)
print("Additive Attention Weights Shape:", additive_attn_weights.shape)  # Expected: (batch_size, seq_length)

# Optionally, print the final representations and attention weights
print("\nFinal Representations:")
print(final_representations)

print("\nAdditive Attention Weights:")
print(additive_attn_weights)
