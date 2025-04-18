import torch
import urllib.request
import re
import tiktoken
from importlib.metadata import version
from tiktoken._educational import *
from torch.utils.data import Dataset, DataLoader
from preprocessing import *
from selfAttention import *
from trainableAttention import *
from causalAttention import *


def main():
    """
    Main entry point for project.
    """
    print(f'PyTorch version: {torch.__version__}')
    print(f'GPU-acceleration via MPS available: {torch.backends.mps.is_available()}')
    print('\n')
    print("Anon is coming.")
    print('\n')

    # UNCOMMENT TO SEE ORIGINAL PREPROCESSOR
    # instantiate the preprocessor
    preprocessor = Preprocess()

    # create token embeddings
    input_embeddings = preprocessor.generate_embeddings()
    print(f'Input embeddings::: \n{input_embeddings}\n')

    print(f'\n\n\n:::::::::SIMPLE SELF-ATTENTION EXAMPLE:::::::::\n\n\n')
    # initiate the simple self-attention demo
    simple_self_attn = SimpleSelfAttention()
    simple_self_attn.generateExample()
    
    print(f'\n\n\n:::::::::TRAINABLE SELF-ATTENTION EXAMPLE:::::::::\n\n\n')
    # initiate the trainable self-attention demo
    trainable_self_attn = TrainableSelfAttention()
    trainable_self_attn.generateExample()

    print(f'\n\n\n:::::::::CAUSAL SELF-ATTENTION EXAMPLE:::::::::\n\n\n')
    # initiate the causal self-attention demo
    causal_self_attn = CausalAttention()
    causal_self_attn.generateExample()
    

if __name__ == "__main__":
    main()