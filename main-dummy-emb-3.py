from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F
import torch.optim as optim

import numpy as np 

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn
import math
from functools import partial
from pathlib import Path
from tqdm import tqdm
#import rich
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import transformers
#import tokenizers
#import datasets
#import zipfile
#from huggingface_hub import hf_hub_download
device = 'cpu'

import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel

from sklearn.preprocessing import label_binarize


print('import complete')
############################################################### data loading starts

table_behavior = pq.read_table('ebnerd_small/validation/behaviors.parquet')
table_history = pq.read_table('ebnerd_small/validation/history.parquet')
table_articles = pq.read_table('ebnerd_small/articles.parquet')
df_behavior = table_behavior.to_pandas()
df_history = table_history.to_pandas()
df_articles = table_articles.to_pandas()

titles = df_articles['title'].values
# get all the words into list
split_words = lambda tit : tit.split(' ')
all_words = list(map(split_words, titles))
words = []
for title in all_words:
    for word in title:
        words.append(word.lower())

all_words = np.unique(np.sort(np.array(words)))

print(all_words.shape)
dummy_dictionary_embedding = {}
for i in range(len(all_words)):
    dummy_dictionary_embedding[all_words[i]] = i

################################### 
dim_emb = 300

user_history_npy = np.load('user_history.npy', allow_pickle=True)
articles_shown_npy = np.load('articles_shown.npy', allow_pickle=True)
articles_clicked_npy = np.load('articles_clicked.npy', allow_pickle=True)

history_limit = 20

class BrowsedCandidateClickedDataset(Dataset):
    def __init__(self, browsed, candidate, clicked):
        self.browsed = browsed
        self.candidate = candidate
        self.clicked = clicked
        
    def __len__(self):
        return len(self.browsed)
    
    def __getitem__(self, index):
        return self.browsed[index][-history_limit:], self.candidate[index], self.clicked[index][0]
    
full_dataset = BrowsedCandidateClickedDataset(user_history_npy, articles_shown_npy, articles_clicked_npy)

batch_size = 32

def custom_collate_fn(batch): 
    browsed, candidate, clicked = zip(*batch)
    return list(browsed), list(candidate), list(clicked)

train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

print('data loaded')
################################################################ make model

#from embed import FastTextEmbeddingLayer

word_count = 27693 + 1 # adding one symbol so that we have a special marking for empty word
MAX_WORDS = 30 # maximum number that is in the title

class MyEmbeddingLayer(nn.Module):
    def __init__(self,emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.emb_torch = nn.Embedding(word_count, emb_dim)
        self.dummy_emb = dummy_dictionary_embedding

    def forward(self,text):
        input_shape = text.shape
        titles = text.flatten() #flatten things so that we look just at the titles
        output = []
        for title in titles:
            words = title.split(" ")
            for word in words:
                output.append(torch.IntTensor([self.dummy_emb[word.lower()]]))

            # all titles need to have the same number of "words" so I just add "empty" words at the end
            for i in range(MAX_WORDS - len(words)):
                output.append(torch.IntTensor([word_count-1]))

        output = torch.stack(output).to(device)
        #print(output.device)
        output = self.emb_torch(output)

        # invert the action of flataning
        output = output.reshape(input_shape + (-1,self.emb_dim))
        return output
    
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
        #print(x.shape)
        input_shape = x.shape
        
        merged_batch_and_titles = x.reshape((-1,) + (input_shape[-2], input_shape[-1]))
        
        attn_output, attn_weights = self.multihead_attn(
            query=merged_batch_and_titles,
            key=merged_batch_and_titles,
            value=merged_batch_and_titles,
            key_padding_mask=attention_mask  # Masks padded tokens if provided
        )
        
        #print('att out',attn_output.shape)
        x = attn_output.reshape(input_shape)

        # Apparently not used in the paper.
        # TODO:
        #   This be an idea to improve the model, maybe bring back with it the normalization.
        # Add residual connections
        # x = x + attn_output

        return x, attn_weights
    
class AdditiveWordAttention(nn.Module):
    def __init__(self, embedding_dimension, additive_vector_dim=200):
        super().__init__()
        self.activation_fn = nn.Tanh()
        self.lin_vw = nn.Linear(in_features=embedding_dimension, out_features=additive_vector_dim)
        self.lin_q = nn.Linear(in_features=additive_vector_dim, out_features=1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, h):
        # lin_vw(h) = V_w × h_i^w + v_w
        # lin_q(act_fn(...)) = q_w^T tanh(...)
        tmp = self.activation_fn(self.lin_vw(h))
        aw = self.lin_q(tmp)
        aw = self.softmax(aw) # exp(...) / SUM exp(...)
        r = aw.transpose(-2,-1) @ h # SUM a_i^w h_i^w
        return r
    
class MyNewsEncoder(nn.Module):
    def __init__(self, embedding_dimension, head_count=10, head_vector_size=30, embedding_dropout=0.3):
        super().__init__()
        #assert embedding_dimension % head_count == 0, "embeding must be divisible by heads"
        self.embedding_dimension = embedding_dimension
        self.embedding = MyEmbeddingLayer(dim_emb) #, dummy_dictionary_embedding)
        #self.embedding = XLMRobertaWordEmbedder()
        #self.embedding = FastTextEmbeddingLayer(embedding_dimension)
        self.embedding_drop = nn.Dropout(embedding_dropout)
        #self.mult_head_att = MultiHeadSelfAttHead(embedding_dimension, head_count, head_vector_size)
        #print(embedding_dimension, head_count)
        self.mult_head_att = PytorchMultiHeadSelfAttHead(embedding_dimension, head_count)
        #print('in word add', head_count, head_vector_size)
        self.add_word_att = AdditiveWordAttention(head_count * head_vector_size)# 16 heads and 16 dimensions each # TODO later change the vector dim to 200

    def forward(self, x): # x is a string of words - title
        
        #input_shape = x.shape
        #print('0_0', input_shape)
        #flatten_titles = x.flatten()
        #print('0_1', flatten_titles.shape)
        #titles_list = flatten_titles.tolist()
        #token_embeddings, attention_mask = self.embedding(titles_list)
        #e_s = token_embeddings.reshape(input_shape + (30, self.embedding_dimension))
        e_s = self.embedding(x)
        e_s = self.embedding_drop(e_s)
        #print('1',e_s.shape)
        
        h, ignore = self.mult_head_att(e_s)
        #print('1_1',h.shape)
        
        r = self.add_word_att(h)
        #print('1_2',r.shape)
        return r.squeeze(dim=-2)
    
class UserEncoder(nn.Module):
    def __init__(self, emb_dimension, user_head_count=10, news_head_count=10, head_vector_size=30):
        super().__init__()
        
        self.news_encoder = MyNewsEncoder(emb_dimension, news_head_count, head_vector_size)
        #self.multi_head_att = MultiHeadSelfAttHead(news_head_count*head_vector_size, user_head_count)
        self.multi_head_att = PytorchMultiHeadSelfAttHead(news_head_count*head_vector_size, user_head_count)
        self.add_news_att = AdditiveWordAttention(user_head_count*head_vector_size)
    
    def forward(self,x):
        
        r = self.news_encoder(x)
        #print('2',r.shape)
        
        h, ignore = self.multi_head_att(r)
        #print('2_1',h.shape)
        
        u = self.add_news_att(h)
        #print('2_2',u.shape)
        
        return u.squeeze(dim=-2)
    
class ClickPredictor(nn.Module):
    #def __init__(self, emb_dimension, user_head_count=16, news_head_count=16, head_vector_size=48):
    def __init__(self, emb_dimension, user_head_count=10, news_head_count=10, head_vector_size=30):
        super().__init__()
        self.userEncoder = UserEncoder(emb_dimension, user_head_count, news_head_count, head_vector_size)
        self.news_encoder = MyNewsEncoder(emb_dimension, news_head_count, head_vector_size)
        
    def forward(self, browsed_news, candidate_news):
        
        u = self.userEncoder(browsed_news)
        u = u.unsqueeze(-2)
        
        r = self.news_encoder(candidate_news)
        
        ŷ = u @ r.transpose(-2, -1) # = u^T r^c
        #ŷ = torch.tensor([torch.dot(u[i], r[i]) for i in range(u.shape[0])])
        
        return ŷ.squeeze(dim=-2)

print('model created')    
############################################################################# training

def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if device == "cuda":
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if device == "cuda":
        return x.cpu().data.numpy()
    return x.data.numpy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dim_emb = 300

model = ClickPredictor(dim_emb)
model.to(device)
full_dataset
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 1
validation_every_steps = 100

step = 0
model.train()

train_accuracies = []
train_loss = []
validation_accuracies = []
validation_loss = []
print('starting_validation')
        
for epoch in range(num_epochs):
    
    train_accuracies_batches = []
    train_loss_batches = []
    
    for browsed, candidate, clicked in train_loader:#[(tmp_dk_input, target)]:#train_loader:#[(dk_input, target)]:#train_loader:
        #print(targets)
        # Forward pass.
        #print('broken',inputs)
        # print('working',target)
        # print('in brow', browsed)
        # print('in brow', np.array(browsed))
        # print('in brow', np.array(browsed).shape)
        # print('in cand', candidate)
        # print('in cand', np.array(candidate))
        #print('in cand', np.array(candidate).shape)
        
        output = model(np.array(browsed), np.array(candidate))#model(np.array(tuple(dk_input)))#model(np.array(inputs))
        #output = model(np.array(browsed))
        
        # Compute loss.
        #print(clicked)
        targ_ind = torch.tensor(clicked).to(device)
        loss = loss_fn(output, targ_ind)
        train_loss_batches.append(loss.cpu().data.numpy())#get_numpy(loss))#.detach().numpy())
        # Clean up gradients from the model.
        optimizer.zero_grad()
        
        # Compute gradients based on the loss from the current batch (backpropagation).
        loss.backward()
        
        # Take one optimizer step using the gradients computed in the previous step.
        optimizer.step()
        
        step += 1
        
        # Compute accuracy.
        #print(output)
        predictions =  torch.argmax(output, dim=-1)#.max(1)[1]
        #print('out:', output)
        #print('predictions:', predictions)
        #print('targets:', targ_ind)
        #print('targ_ind', targ_ind)
        #print('predictions', predictions)
        #print(targ_ind.device)
        #print(predictions.device)

        y_true = targ_ind.cpu().data.numpy()
        y_pred = F.softmax(output, dim=-1).cpu().data.numpy()
        batch_classes = np.unique(y_true)
        y_true_binarized = label_binarize(y_true, classes=batch_classes)
        #print(y_true.shape)
        #print(y_true_binarized.shape)
        #print(y_pred.shape)
        #print(y_pred[:,batch_classes])
        calculated_acc = roc_auc_score(y_true_binarized, y_pred[:, batch_classes], multi_class='ovr')
        #calculated_acc = accuracy_score(targ_ind.cpu().data.numpy(), predictions.cpu().data.numpy())
        train_accuracies_batches.append(calculated_acc)
        
        
        if step % validation_every_steps == 0:
            
            # Append average training accuracy to list.
            train_accuracies.append(np.mean(train_accuracies_batches))
            train_loss.append(np.mean(train_loss_batches))
            
            train_accuracies_batches = []
            train_loss_batches = []
        
            # Compute accuracies on validation set.
            # validation_accuracies_batches = []
            # with torch.no_grad():
            #     model.eval()
            #     for inputs, targets in validation_loader:
            #         output = model(inputs)
            #         loss = loss_fn(output, targets.float())

            #         predictions = output.max(1)[1]
            #         targ_ind = targets.max(1)[1]
                    
            #         # Multiply by len(x) because the final batch of DataLoader may be smaller (drop_last=False).
            #         validation_accuracies_batches.append(accuracy_score(targ_ind, predictions) * len(inputs))

            #     model.train()
                
            # # Append average validation accuracy to list.
            # validation_accuracies.append(np.sum(validation_accuracies_batches) / len(validation_dataset))
     
            print(f"Step {step:<5}   training accuracy: {train_accuracies[-1]}, loss: {train_loss[-1]}")
            #print(f"             validation accuracy: {validation_accuracies[-1]}")
            

print("Finished training.")

folder = 'dummy3/'

torch.save(model.state_dict(), folder + 'trained_model')

with open(folder + 'train_acc.npy', 'wb') as f:
    np.save(f,np.array(train_accuracies))

with open(folder + 'train_loss.npy', 'wb') as f:
    np.save(f, np.array(train_loss))
    

######################################### computing the validation

main_table = df_behavior[['article_ids_inview','article_ids_clicked','user_id']]
joined_table = main_table.join(df_history[['user_id', 'article_id_fixed']].set_index('user_id'), on='user_id', validate='many_to_one')

K = 4
def remove_clicked(row):
    index_of_clicked_one = np.where(row['article_ids_inview'] == row['article_ids_clicked'][0])
    indexes_of_not_clicked = np.delete(row['article_ids_inview'], index_of_clicked_one)
    indexes_of_not_clicked_suffled = np.random.choice(indexes_of_not_clicked, size=(K), replace=False) # now we have list of K = 4 things
    indexes_of_all = np.concatenate((indexes_of_not_clicked_suffled, [row['article_ids_clicked'][0]]), axis=0) # merge random no selected ones and the selected one
    np.random.shuffle(indexes_of_all) # suffle them
    correct_index = np.where(indexes_of_all == row['article_ids_clicked'][0]) # get the index - used as label
    return [indexes_of_all, correct_index[0]]
  
joined_table[['articles_input_ids', 'articles_correct_idx']] = joined_table.apply(remove_clicked, axis=1, result_type='expand')

article_map = df_articles.set_index('article_id') # this make a significant speedup in the following method

def from_ids_arr_to_article_title_arr(ids_arr):
    return article_map.loc[ids_arr]['title'].values
  
articles_shown = joined_table[['articles_input_ids']][:].map(from_ids_arr_to_article_title_arr) # shown articles

articles_clicked = joined_table['articles_correct_idx'] # index of selected article

article_history = joined_table[['article_id_fixed']][:].map(from_ids_arr_to_article_title_arr) # history of articles shown

max_len = article_history['article_id_fixed'].apply(len).max()
def pad_list(row):
    padded_row = np.append(row, [''] * (max_len - len(row)))
    return np.array(padded_row)
  
article_history['article_id_fixed_padded'] = article_history['article_id_fixed'].apply(pad_list)

user_history_npy = article_history['article_id_fixed_padded'].values
articles_shown_npy = articles_shown['articles_input_ids'].values
articles_clicked_npy = articles_clicked.values

validation_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

full_dataset
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, roc_curve
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import cycle

# Set the model to evaluation mode
model.eval()

# Define the percentage of data to use
data_percentage = 0.01  # Use 10% of the dataset

# Total number of batches for progress reporting
total_batches = len(validation_loader)
max_batches = int(total_batches * data_percentage)  # Limit to 10% of the batches

# Tracking metrics
validation_accuracies = []
validation_auc_scores = []
validation_loss = []
batch_indices = []
all_y_true = []
all_y_pred = []

print("Starting validation...")

with torch.no_grad():  # Disable gradient computation
    for batch_idx, (browsed, candidate, clicked) in enumerate(validation_loader):
        if batch_idx >= max_batches:  # Stop after processing 10% of the dataset
            break

        # Progress output
        print(f"Processing batch {batch_idx + 1}/{total_batches} (up to {max_batches})...", end="\r")

        # Forward pass
        output = model(np.array(browsed), np.array(candidate))

        # Compute loss
        targ_ind = torch.tensor(clicked).to(device)
        loss = loss_fn(output, targ_ind)
        validation_loss.append(loss.cpu().data.numpy())
        batch_indices.append(batch_idx + 1)

        # Compute predictions and metrics
        predictions = torch.argmax(output, dim=-1).cpu().numpy()
        y_true = targ_ind.cpu().numpy()
        y_pred = F.softmax(output, dim=-1).cpu().numpy()

        # Collect all true and predicted probabilities for aggregate metrics
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # Accuracy
        batch_accuracy = accuracy_score(y_true, predictions)
        validation_accuracies.append(batch_accuracy)

        # AUC
        batch_classes = np.unique(y_true)
        y_true_binarized = label_binarize(y_true, classes=batch_classes)
        if len(batch_classes) > 1:  # Avoid AUC computation for a single class
            batch_auc = roc_auc_score(y_true_binarized, y_pred[:, batch_classes], multi_class='ovr')
            validation_auc_scores.append(batch_auc)
        else:
            validation_auc_scores.append(float('nan'))  # Skip if AUC is not computable

# Summarize results

with open(folder + 'f1.npy','wb') as f:
  np.save(f,np.array(validation_loss))
  
with open(folder + 'f2.npy','wb') as f:
  np.save(f,np.array(batch_indices))
  
with open(folder + 'f3.npy','wb') as f:
  np.save(f,np.array(validation_accuracies))
  
with open(folder + 'f4.npy','wb') as f:
  np.save(f,np.array(validation_auc_scores))
  
with open(folder + 'f5.npy','wb') as f:
  np.save(f,np.array(all_y_true))
  
with open(folder + 'f6.npy','wb') as f:
  np.save(f,np.array(all_y_pred))
