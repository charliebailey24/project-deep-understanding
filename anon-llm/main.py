import torch
import urllib.request
import re

def tokenize(text):
    """
    Split raw text into a list of tokens.
    """
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [tok.strip() for tok in tokens if tok.strip()]
    return tokens

def preprocess(raw_text):
    """
    Tokenizes the raw text and converts is to a vector embedding.
    """
    # tokenize the text
    tokens = tokenize(raw_text)

    # generate token ids
    unique_tokens = sorted(set(tokens))
    # vocab_size  = len(unique_tokens)
    token_ids = {tok:int for int, tok in enumerate(unique_tokens)}
    return token_ids

def get_sample_text():
    """
    Retrieves the raw text of 'The Verdict' from the BaLLM repository.  
    """
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    filename, _ = urllib.request.urlretrieve(url, file_path)
    with open(filename, "r", encoding="utf-8") as f:
        raw_text = f.read()
    return raw_text

def main():
    """
    Main entry point for project.
    """
    print(f'PyTorch version: {torch.__version__}')
    print(f'GPU-acceleration via MPS available: {torch.backends.mps.is_available()}')
    print("Anon is coming.")
    print('\n')
    raw_text = get_sample_text()
    token_ids = preprocess(raw_text)
    print(token_ids)

if __name__ == "__main__":
    main()