import fasttext
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DO NOT LOAD THE MODEL MORE THEN ONCE, IT TAKES A LOT OF RAM
fasttext_model = fasttext.load_model('cc.da.300.bin')

MAX_WORDS = 30
class FastTextEmbeddingLayer(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        #self.emb_torch = nn.Embedding(word_count, emb_dim)
        self.embedding_fasttext = fasttext_model # this is to avoid loading the model every time we create a model, as it is very RAM hungry
        
    def forward(self,text):
        input_shape = text.shape
        titles = text.flatten() #flatten things so that we look just at the titles
        output = []
        for title in titles:
            words = title.split()
            for word in words:
                output.append(torch.from_numpy(self.embedding_fasttext.get_word_vector(word)))
                
            # all titles need to have the same number of "words" so I just add "empty" words at the end
            for i in range(MAX_WORDS - len(words)): 
                output.append(torch.zeros(300)) # this is vector for string with space
        
        output = torch.stack(output).to(device)
        #print(output.device)
        #output = self.emb_torch(output)
        
        # invert the action of flataning
        output = output.reshape(input_shape + (-1,self.emb_dim)) 
        return output