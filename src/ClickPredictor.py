import torch.nn as nn
from UserEncoder import UserEncoder
from NewsEncoder import NewsEncoder


class ClickPredictor(nn.Module):
    def __init__(self, emb_dimension, user_head_count=1, news_head_count=1):
        super().__init__()
        self.userEncoder = UserEncoder(emb_dimension, user_head_count, news_head_count)
        self.news_encoder = NewsEncoder(emb_dimension, news_head_count)

    def forward(self, browsed_news, candidate_news):

        u = self.userEncoder(browsed_news)
        u = u.unsqueeze(-2)

        r = self.news_encoder(candidate_news)

        ŷ = u @ r.transpose(-2, -1) # = u^T r^c
        #ŷ = torch.tensor([torch.dot(u[i], r[i]) for i in range(u.shape[0])])

        return ŷ.squeeze(dim=-2)
