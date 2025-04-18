import torch
import urllib.request
import re
import tiktoken
from importlib.metadata import version
from tiktoken._educational import *
from torch.utils.data import Dataset, DataLoader
from preprocessing import *
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    """
    A simple self-attention layer that computes attention scores and context vectors.
    The nn.Module class is a fundamental building block for PyTorch models.
    This class is used to compute attention scores and context vectors based on input embeddings.
    """
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        # compute query, key and value vectors
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        # compute attention scores
        attn_scores = queries @ keys.T

        # apply softmax to the attention scores
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)

        # compute context vector
        context_vec = attn_weights @ values

        return context_vec
    
class SelfAttention_v2(nn.Module):
    """
    A more complex self-attention layer that computes attention scores and context vectors.
    The nn.Module class is a fundamental building block for PyTorch models.
    This class is used to compute attention scores and context vectors based on input embeddings.
    """
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.attn_weights = None

    def get_attn_weights(self):
        """
        Returns the attention weights.
        """
        return self.attn_weights

    def forward(self, x):
        # compute query, key and value vectors
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # compute attention scores
        attn_scores = queries @ keys.T

        # apply softmax to the attention scores
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        self.attn_weights = attn_weights

        # compute context vector
        context_vec = attn_weights @ values

        return context_vec
    

class TrainableSelfAttention():

    def generateExample(self):
        '''
        Example of computing the attention weights step by step.
        '''

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

        x_2 = inputs[1]

        # why are input and output shapes different?
        d_in = inputs.shape[1]
        d_out = 2

        # initialize the query, key and value weight matrices
        torch.manual_seed(123)
        W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
        W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

        # compute single query, key and value vectors
        query_2 = x_2 @ W_query
        key_2 = x_2 @ W_key
        value_2 = x_2 @ W_value
        print(f'Example of the query vector for the input embedding of the word "journey"::: \n{query_2}\n')

        # use the dot product to compute all key and value vectors
        keys = inputs @ W_key
        values = inputs @ W_value

        print(f'Shape of keys matrix::: \n{keys.shape}\n')
        print(f'Shape of values matrix::: \n{values.shape}\n')

        # attention score for the word "journey" with itself
        keys_2 = keys[1]
        attn_score_22 = query_2.dot(keys_2)
        print(f'Attention score for the word "journey" with itself::: \n{attn_score_22}\n')

        # generalize to all attention scores
        attn_scores_2 = query_2 @ keys.T
        print(f'Attention scores for the word "journey" with all other words::: \n{attn_scores_2}\n')
        # get the shape of the attention scores
        print(f'Shape of attention scores::: \n{attn_scores_2.shape}\n')

        # apply softmax to the attention scores
        d_k = keys.shape[-1]
        attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
        print(f'Attention weights for the word "journey" with all other words::: \n{attn_weights_2}\n')

        # compute the context vector
        context_2 = attn_weights_2 @ values
        print(f'Context vector for the word "journey"::: \n{context_2}\n')

        print(f'\n\n\n:::::::::SELF-ATTENTION V1 TEST:::::::::\n\n\n')
        # instantiate the self-attention layer
        torch.manual_seed(123)
        self_attn_v1 = SelfAttention_v1(d_in, d_out)
        print(f'All context vectors for "Your journey begins with a single step"::: \n{self_attn_v1(inputs)}\n')

        print(f'\n\n\n:::::::::SELF-ATTENTION V2 TEST:::::::::\n\n\n')
        # instantiate the self-attention layer
        torch.manual_seed(789)
        self_attn_v2 = SelfAttention_v2(d_in, d_out)
        print(f'All context vectors for "Your journey begins with a single step"::: \n{self_attn_v2(inputs)}\n')


