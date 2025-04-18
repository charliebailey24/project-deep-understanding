import torch
import urllib.request
import re
import tiktoken
from importlib.metadata import version
from tiktoken._educational import *
from torch.utils.data import Dataset, DataLoader
from preprocessing import *


class SimpleSelfAttention():

    def generateExample(self):
        '''
        Work through self-attention mechanism using text:
        "Your journey begins with one step."
        '''
        # dataloader did not work with such a small input
        # used example tensor from BaLLM
        print(f'Sample Text::: "Your journey begins with one step"\n')

        inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your    (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )

        print(f'Input embeddings::: \n{inputs}\n')
        query = inputs[1]
        print(f'Query token input embedding::: \n{query}\n')
        attn_scores_x2 = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            attn_scores_x2[i] = torch.dot(x_i, query)
        
        print(f'Attention Scores::: \n{attn_scores_x2}\n')

        # use the PyTorch softmax function to get the 
        # normalized attention weights for the second input token
        attn_weights_x2 = torch.softmax(attn_scores_x2, dim=0)
        print(f'Attentions weights::: \n{attn_weights_x2}\n')
        # confirm weights sum to 1
        print(f'Confirm weights sum to 1::: \n{attn_weights_x2.sum()}\n')

        # what is going on here??
        context_vec_x2 = torch.zeros(query.shape)
        for i, x_i in enumerate(inputs):
            context_vec_x2 += attn_weights_x2[i] * x_i
        
        print(f'Context vec for x2 input embedding::: \n{context_vec_x2}\n')

        attn_scores = torch.empty(6, 6)
        for i, x_i in enumerate(inputs):
            for j, x_j in enumerate(inputs):
                attn_scores[i, j] = torch.dot(x_i, x_j)

        print(f'For loop attention scores::: \n{attn_scores}\n')

        attn_scores_matrix = inputs @ inputs.T

        print(f'Matrix multiplication attention scores::: \n{attn_scores_matrix}\n')

        # normalize each row so the values sum to 1
        attn_weights = torch.softmax(attn_scores_matrix, dim=-1)

        print(f'All attention weights::: \n{attn_weights}\n')

        # confirm all rows sum to 1
        print(f'All rows sums::: \n{attn_weights.sum(dim=-1)}\n')

        # use the attention weights to compute all context vectors
        all_context_vecs = attn_weights @ inputs
        print(f'All context vectors::: \n{all_context_vecs}\n')