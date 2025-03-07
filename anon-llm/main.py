import torch
import urllib.request
import re
import tiktoken
from importlib.metadata import version
from tiktoken._educational import *
from torch.utils.data import Dataset, DataLoader

def educational_bpe():
    """
    FOR EDUCATIONAL PURPOSES ONLY!
    Visualization tool for how the GPT-4 encoder encodes text.
    Prints directly to the terminal.
    """
    enc = train_simple_encoding()
    enc = SimpleBytePairEncoding.from_tiktoken("cl100k_base")
    enc.encode("hello world, i am anon")

class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

class Preprocess:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.sample_text = self.get_sample_text()

    def get_sample_text(self):
        """
        Retrieves the raw text of 'The Verdict' from the BaLLM repository.  
        """
        url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
        file_path = "the-verdict.txt"
        filename, _ = urllib.request.urlretrieve(url, file_path)
        with open(filename, "r", encoding="utf-8") as f:
            raw_text = f.read()
        return raw_text

    def generate_pairs(self, enc_text, context_size=4):
        """
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

    def bpe_tokenize(self, text):
        """
        Split raw text into a list of tokens using byte-pair encoding.
        """
        # t_version = version('tiktoken')
        # print(f"tiktoken version::: {t_version}")
        # tokenizer = tiktoken.get_encoding("gpt2")
        enc_text = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return enc_text


class SimplePreprocess():
    def tokenize(self, text):
        """
        DEPRECATED.
        Split raw text into a list of tokens using Regex.
        """
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [tok.strip() for tok in tokens if tok.strip()]
        return tokens

    def preprocess(self, raw_text):
        """
        DEPRECATED.
        Tokenizes the raw text and converts is to a vector embedding.
        """
        # tokenize the text
        tokens = self.tokenize(raw_text)

        # generate token ids
        unique_tokens = sorted(set(tokens))
        # vocab_size  = len(unique_tokens)
        token_ids = {tok:int for int, tok in enumerate(unique_tokens)}
        return token_ids


def main():
    """
    Main entry point for project.
    """
    print(f'PyTorch version: {torch.__version__}')
    print(f'GPU-acceleration via MPS available: {torch.backends.mps.is_available()}')
    print("Anon is coming.")
    print('\n')

    preprocessor = Preprocess()
    # get the sample text
    raw_text = preprocessor.get_sample_text()
    # tokenize text using BPE
    enc_text = preprocessor.bpe_tokenize(raw_text)
    # generate the input-target pairs
    preprocessor.generate_pairs(enc_text)

if __name__ == "__main__":
    main()