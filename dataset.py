import os
import urllib.request
import tarfile
import torch
from torch.utils.data import IterableDataset
import sentencepiece as spm

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

def train_sentencepiece_model(text_file, vocab_size=32000, model_prefix="spm_wt103"):
    if not os.path.exists(f"{model_prefix}.model"):
        print("Training SentencePiece model...")
        spm.SentencePieceTrainer.train(
            input=text_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

class SentencePieceWikiTextDataset(IterableDataset):
    def __init__(self, file_path, sp_processor, tgt_len):
        self.file_path = file_path
        self.sp = sp_processor
        self.tgt_len = tgt_len

    def __iter__(self):
        buffer = []
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text: continue
                buffer.extend(self.sp.encode_as_ids(text))

                while len(buffer) >= self.tgt_len + 1:
                    chunk = buffer[:self.tgt_len + 1]
                    buffer = buffer[self.tgt_len:]
                    yield torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)