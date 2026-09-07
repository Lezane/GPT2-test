import os
import urllib.request
import tarfile
import torch
from torch.utils.data import IterableDataset
from tokenizers import ByteLevelBPETokenizer

def download_wikitext103(dataset_dir="wikitext-103", tgz_path="wikitext-103.tgz"):
    if not os.path.exists(dataset_dir):
        print("Downloading Dataset...")
        urllib.request.urlretrieve("https://s3.amazonaws.com/fast-ai-nlp/wikitext-103.tgz", tgz_path)
        with tarfile.open(tgz_path, 'r:gz') as tar_ref:
            if hasattr(tarfile, 'data_filter'):
                tar_ref.extractall(".", filter='data')
            else:
                tar_ref.extractall(".")
    return os.path.join(dataset_dir, "train.csv")

def train_bpe_model(text_file, vocab_size=32000, model_prefix="bpe_wt103"):
    vocab_file = f"{model_prefix}-vocab.json"
    merges_file = f"{model_prefix}-merges.txt"
    
    # Train only if the tokenizer files don't already exist
    if not (os.path.exists(vocab_file) and os.path.exists(merges_file)):
        print("Training BPE model...")
        tokenizer = ByteLevelBPETokenizer()
        
        # Train BPE using your specific dataset
        tokenizer.train(
            files=[text_file],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<s>", "</s>"]
        )
        tokenizer.save_model(".", model_prefix)
    
    # Load and return the tokenizer
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    return tokenizer

class BPEWikiTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, tgt_len):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.tgt_len = tgt_len

    def __iter__(self):
        buffer = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text: continue
                
                # tokenizer.encode() returns an Encoding object; .ids extracts the list of integer token IDs
                buffer.extend(self.tokenizer.encode(text).ids)

                while len(buffer) >= self.tgt_len + 1:
                    chunk = buffer[:self.tgt_len + 1]
                    buffer = buffer[self.tgt_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
