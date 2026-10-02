import os
import urllib.request
import tarfile
from collections import Counter
import torch
from torch.utils.data import IterableDataset
from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = ("<pad>", "<unk>", "<s>", "</s>")

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
            special_tokens=list(SPECIAL_TOKENS)
        )
        tokenizer.save_model(".", model_prefix)
    
    # Load and return the tokenizer
    tokenizer = ByteLevelBPETokenizer(vocab_file, merges_file)
    return tokenizer

def iter_encoded_lines(file_path, tokenizer):
    """Use the same text preprocessing for frequency counts and training."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                yield tokenizer.encode(text).ids

def build_frequency_groups(file_path, tokenizer, major_vocab_frac=0.95):
    """Split observed, non-special types by final corpus frequency.

    The majority contains floor(major_vocab_frac * eligible_types) types,
    clamped to leave at least one type in each group. Frequency ties are
    broken by token ID. Special and unobserved types belong to neither group.
    Returns CPU tensors: counts, majority membership, minority membership.
    """
    if not 0.0 < major_vocab_frac < 1.0:
        raise ValueError("major_vocab_frac must be strictly between 0 and 1")
    frequencies = Counter()
    for ids in iter_encoded_lines(file_path, tokenizer):
        frequencies.update(ids)

    vocab_size = tokenizer.get_vocab_size()
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for token_id, count in frequencies.items():
        counts[token_id] = count
    special_ids = {tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    ranked_ids = sorted(
        (i for i in frequencies if i not in special_ids),
        key=lambda i: (-frequencies[i], i),
    )
    if len(ranked_ids) < 2:
        raise ValueError("Need at least two observed, non-special token types")
    n_major = max(1, min(len(ranked_ids) - 1, int(major_vocab_frac * len(ranked_ids))))
    is_major = torch.zeros(vocab_size, dtype=torch.bool)
    is_minor = torch.zeros(vocab_size, dtype=torch.bool)
    is_major[ranked_ids[:n_major]] = True
    is_minor[ranked_ids[n_major:]] = True
    return counts, is_major, is_minor

class BPEWikiTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, tgt_len):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.tgt_len = tgt_len

    def __iter__(self):
        buffer = []
        for ids in iter_encoded_lines(self.file_path, self.tokenizer):
            buffer.extend(ids)
            while len(buffer) >= self.tgt_len + 1:
                chunk = buffer[:self.tgt_len + 1]
                buffer = buffer[self.tgt_len:]
                yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)
