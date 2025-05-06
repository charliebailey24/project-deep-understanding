import torch
import re
import os
import tiktoken
from tiktoken._educational import *
from torch.utils.data import Dataset, DataLoader

# inherits from PyTorch Dataset class
# defines how individual rows are fetched from dataset
class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # tokenize the entire text
        token_ids = tokenizer.encode(text)

        # use a sliding window to create overlapping chunks of size max length
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    # return total num rows in the dataset
    def __len__(self):
        return len(self.input_ids)
    
    # return a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class Preprocess():
    def __init__(self, text=None, batch_size=8, max_length=4, stride=4,
                 shuffle=False, drop_last=True, num_workers=0):
        # get sample text if no text passed in
        self.text = text if text else self.get_sample_text()
        self.batch_size = batch_size
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.dataloader = self.create_dataloader_v1(text=self.text,
                                                    batch_size=batch_size, max_length=max_length,
                                                    stride=stride, shuffle=shuffle,
                                                    drop_last=drop_last, num_workers=num_workers)

    def get_sample_text(self):
        """
        Retrieves the raw text of 'The Verdict' from a local file in the ../data folder.
        """
        
        # path to the file in the data directory one level up
        file_path = os.path.join("..", "data", "the-verdict.txt")
        
        # Read the file contents
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            return raw_text
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            print("Make sure 'the-verdict.txt' exists in the ../data directory")
            return ""
    
    def create_dataloader_v1(self, text, batch_size, max_length, stride,
                         shuffle, drop_last=True, num_workers=0):
        """
        Builds a dataloader using the GPT-2 tokenizer encodings and PyTorch DataLoader class.
        Standard training input size is at least 256.
        Stride determines number of positions input shifts across batches (sliding window effect)
        """
        # initialize tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # create dataset
        dataset = GPTDatasetV1(text=text, tokenizer=tokenizer, max_length=max_length, stride=stride)

        # build dataloader from PyTorch DataLoader class
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader

    def generate_embeddings(self, vocab_size=50257, output_dim=256):
        """
        Generates the positional embeddings for a given set of token ids.
        """
        # print the current text
        print(f'Start of current text::: {self.text[:25]}')
        
        # load data 
        data_iter = iter(self.dataloader)

        # generate input and target tensors
        # need to figure out what to do with targets
        inputs, targets = next(data_iter)
        print(f'input shape::: {inputs.shape}\n')
        print(f'inputs from dataloader::: \n{inputs}\n')
        print(f'targets from dataloader::: \n{targets}\n')

        # create the token embedding layer
        token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

        # embed the input tokens into the embedding layer
        token_embeddings = token_embedding_layer(inputs)
        print(f'token embeddings shape::: {token_embeddings.shape}\n')

        # set context length
        context_length = self.max_length

        # create positional embedding vector
        pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
        pos_embeddings = pos_embedding_layer(torch.arange(context_length))
        print(f'positional embeddings shape::: {pos_embeddings.shape}\n')

        # add token embeddings to positional embeddings
        # how does the + operator work in PyTorch?
        input_embeddings = token_embeddings + pos_embeddings
        print(f'input embeddings shape::: {pos_embeddings.shape}\n')

        return input_embeddings

    def bpe_tokenize(self, text):
        """
        Split raw text into a list of tokens using byte-pair encoding.
        """
        # t_version = version('tiktoken')
        # print(f"tiktoken version::: {t_version}")
        # tokenizer = tiktoken.get_encoding("gpt2")
        enc_text = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return enc_text
    
    def generate_pairs(self, enc_text, context_size=4):
        """
        FOR DEMONSTRATION PURPOSES ONLY!!
        Generates the input-target pairs using a sliding window of size 1.
        """
        enc_sample = enc_text[50:]
        context_size = 4
        x = enc_sample[:context_size]
        y = enc_sample[1:context_size+1]
        print(f'x: {x}')
        print(f'y:      {y}')

        for i in range(1, context_size+1):
            context = enc_sample[:i]
            desired = enc_sample[i]
            print(context, "---->", desired)
        print('\n')
        for j in range(1, context_size+1):
            context = enc_sample[:j]
            desired = enc_sample[j]
            print(self.tokenizer.decode(context), "---->", self.tokenizer.decode([desired]))

#################### FOR EDUCATIONAL PURPOSES ONLY!! ########################
class SimplePreprocess():
    def tokenize(self, text):
        """
        Split raw text into a list of tokens using Regex.
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [tok.strip() for tok in tokens if tok.strip()]
        return tokens

    def preprocess(self, raw_text):
        """
        Tokenizes the raw text and converts is to a vector embedding.
        """
        # tokenize the text
        tokens = self.tokenize(raw_text)

        # generate token ids
        unique_tokens = sorted(set(tokens))
        # vocab_size  = len(unique_tokens)
        token_ids = {tok:int for int, tok in enumerate(unique_tokens)}
        return token_ids

class Visualize():
    def educational_bpe():
        """
        Visualization tool for how the GPT-4 encoder encodes text.
        Prints directly to the terminal.
        """
        enc = train_simple_encoding()
        enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
        enc.encode("hello world, i am anon")

    def educational_sliding_window():
        preprocessor = Preprocess()
        # get the sample text
        raw_text = preprocessor.get_sample_text()
        # tokenize text using BPE
        enc_text = preprocessor.bpe_tokenize(raw_text)
        # generate the input-target pairs
        preprocessor.generate_pairs(enc_text)