import torch.nn as nn
from NewsEncoder import NewsEncoder
from AdditiveWordAttention import AdditiveWordAttention
from PytorchMultiHeadSelfAttHead import PytorchMultiHeadSelfAttHead
from ManualMultiHeadSelfAttHead import ManualMultiHeadSelfAttHead

class UserEncoder(nn.Module):
    def __init__(self, emb_dimension, user_head_count=1, news_head_count=1):
        super().__init__()

        self.news_encoder = NewsEncoder(emb_dimension, news_head_count)
        self.multi_head_att = ManualMultiHeadSelfAttHead(emb_dimension, user_head_count)
        self.add_news_att = AdditiveWordAttention(emb_dimension, emb_dimension)

    def forward(self,x):

        r = self.news_encoder(x)
        print('2',r.shape)

        h = self.multi_head_att(r)
        print('2_1',h.shape)

        u = self.add_news_att(h)
        print('2_2',u.shape)

        return u.squeeze(dim=-2)
