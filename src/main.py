import torch
from NewsEncoderModel import NewsEncoderModel

# Initialize the model
encoder = NewsEncoderModel()

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
encoder.eval()

# Example titles
titles = [
    "Rockets Ends 2018 with a Win",
    "Another News Title",
    "Breaking News: NFL Championship Highlights",
    "Today in Technology: New Innovations Released"
]

# Generate embeddings and representations
with torch.no_grad():
    enhanced_embeddings, self_attn_weights, final_representations, additive_attn_weights = encoder(titles)

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
