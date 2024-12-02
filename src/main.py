import torch
import torch.nn as nn
import numpy as np
from ClickPredictor import ClickPredictor
from NewsEncoder import NewsEncoder
from UserEncoder import UserEncoder

# Initialize the model
# model_news = NewsEncoder(768, head_count=10)
model_users = UserEncoder(768, news_head_count=16)
model_click = ClickPredictor(768)


# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_news.to(device)
# model_news.eval()
model_users.to(device)
model_users.eval()
model_click.to(device)
model_click.eval()

# Example titles
titles_input = np.array(["Natascha var ikke den første", "Kun Star Wars tjente mere", "Luderne flytter på landet"])

# # Generate embeddings and representations
# with torch.no_grad():
#     enhanced_embeddings, self_attn_weights, final_representations, additive_attn_weights = encoder(titles_input)

# # Print output shapes
# print("Enhanced Embedding Shape:", enhanced_embeddings.shape)  # Expected: (batch_size, seq_length, hidden_size)
# print("Self-Attention Weights Shape:", self_attn_weights.shape)  # Expected: (batch_size, num_heads, seq_length, seq_length)
# print("Final Representations Shape:", final_representations.shape)  # Expected: (batch_size, hidden_size)
# print("Additive Attention Weights Shape:", additive_attn_weights.shape)  # Expected: (batch_size, seq_length)

# # Optionally, print the final representations and attention weights
# print("\nFinal Representations:")
# print(final_representations)

# print("\nAdditive Attention Weights:")
# print(additive_attn_weights)


dk_input = np.array([["Natascha var ikke den første", "Kun Star Wars tjente mere", "Luderne flytter på landet"],['Cybersex: Hvornår er man utro?','Kniven for struben-vært får selv kniven','Willy Strube har begået selvmord']]) # fist samples to be used

results = model_users(dk_input)
print(results.shape)

# loss_fn = nn.L1Loss()
# optimizer = optim.Adam(model_news.parameters(), lr=1e-4)
