import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import urllib.request
import re
import tiktoken
from tiktoken._educational import *
from importlib.metadata import version
from preprocessing import *
from trainableAttention import *

class CausalAttention_v1(nn.Module):
    """
    A simple causal self-attention layer that computes attention scores and context vectors.
    """
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

class CausalAttention():

    def generateExample(self):
        """
        Work through causal attention mechanism using text:
        "Your journey begins with one step."
        """
        # dataloader did not work with such a small input
        # used example tensor from BaLLM

        inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your    (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]] # step     (x^6)
        )

        d_in = inputs.shape[1]
        d_out = 2

        # instantiate the trainable self-attention class
        torch.manual_seed(789)
        self_attn_v2 = SelfAttention_v2(d_in, d_out)
        queries = self_attn_v2.W_query(inputs)
        keys = self_attn_v2.W_key(inputs)
        attn_scores = queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        print(f'Attention weights::: \n{attn_weights}\n')

        # mask the attention weights above the diagonal
        context_length = attn_weights.shape[0]
        mask = torch.tril(torch.ones(context_length, context_length))
        print(f'Mask::: \n{mask}\n')

        # multiply the attention weights by the mask
        masked_attn_weights = attn_weights * mask
        print(f'Masked attention weights::: \n{masked_attn_weights}\n')

        # re-normalize the masked attention weights
        row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
        masked_simple_norm = masked_attn_weights / row_sums
        print(f'Masked attention weights normalized::: \n{masked_simple_norm}\n')

        # use negative infinity to mask the attention weights
        neg_inf_mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked_attn_scores = attn_scores.masked_fill(neg_inf_mask.bool(), -torch.inf)
        print(f'Masked attention scores::: \n{masked_attn_scores}\n')

        # compute attention weights using the masked attention scores
        masked_attn_weights2 = torch.softmax(masked_attn_scores / d_k**0.5, dim=-1)
        print(f'Masked attention weights using negative infinity::: \n{masked_attn_weights2}\n')

        # example of dropout
        torch.manual_seed(123)
        dropout = nn.Dropout(p=0.5)
        example = torch.ones(6,6)
        print(f'Example dropout::: \n{dropout(example)}\n')

        # apply dropout to the attention weights
        print(f'Attention weights after dropout::: \n{dropout(masked_attn_weights2)}\n')

        # create larger batch size example
        batch = torch.stack((inputs, inputs), dim=0)
        print(f'Batch shape::: {batch.shape}\n')

        # test the causal attention layer
        context_length = batch.shape[1]
        causal_attn = CausalAttention_v1(d_in, d_out, context_length, dropout=0.5)
        context_vecs = causal_attn(batch)
        print(f'Context vectors shape::: {context_vecs.shape}\n')
        print(f'Context vectors::: \n{context_vecs}\n')




        

