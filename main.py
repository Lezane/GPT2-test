import os
import random
import numpy as np
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local modules
import config as cfg
from dataset import download_wikitext103, train_bpe_model, BPEWikiTextDataset, build_frequency_groups
from model import SimpleGPT
from optimizer import configure_adamw, configure_sgd, configure_muon

def set_seed(seed):
    """Ensures exact reproducible initialization between AdamW, SGD, and Muon runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_experiment(optim_type, train_file, tokenizer, vocab_size, is_major, is_minor):
    # 1. Force the exact same random seed right before initializing the model weights
    set_seed(cfg.SEED)
    
    if optim_type == "AdamW":
        batch_size = cfg.ADAM_TRAIN_BATCH_SIZE
        grad_acc_steps = cfg.ADAM_GRADIENT_ACC_STEPS
    elif optim_type == "SGD":
        batch_size = cfg.SGD_TRAIN_BATCH_SIZE
        grad_acc_steps = cfg.SGD_GRADIENT_ACC_STEPS
    elif optim_type == "Muon":
        batch_size = cfg.MUON_TRAIN_BATCH_SIZE
        grad_acc_steps = cfg.MUON_GRADIENT_ACC_STEPS
    else:
        raise ValueError(f"Unknown optimizer: {optim_type}")

    # Initialize the new BPE Iterable Dataset
    train_dataset = BPEWikiTextDataset(train_file, tokenizer, cfg.TARGET_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    print(f"\nInitializing Model for {optim_type} run...")
    model = SimpleGPT(
        vocab_size=vocab_size, emb_dim=cfg.EMB_DIM,
        num_heads=cfg.NUM_HEADS, depth=cfg.DEPTH
    ).to(cfg.DEVICE)

    if optim_type == "AdamW":
        optimizer = configure_adamw(
            model, weight_decay=cfg.ADAM_WEIGHT_DECAY, learning_rate=cfg.ADAM_LEARNING_RATE,
            betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2), eps=cfg.ADAM_EPS
        )
    elif optim_type == "SGD":
        optimizer = configure_sgd(
            model, learning_rate=cfg.SGD_LEARNING_RATE, momentum=cfg.SGD_MOMENTUM
        )
    elif optim_type == "Muon":
        optimizer = configure_muon(
            model, 
            muon_lr=cfg.MUON_LEARNING_RATE, 
            muon_momentum=cfg.MUON_MOMENTUM,
            adamw_lr=cfg.ADAM_LEARNING_RATE, 
            adamw_betas=(cfg.ADAM_BETA1, cfg.ADAM_BETA2), 
            adamw_eps=cfg.ADAM_EPS, 
            adamw_wd=cfg.ADAM_WEIGHT_DECAY
        )

    # Use reduction='none' so we can split unreduced losses by vocabulary subsets mathematically safely
    criterion_none = nn.CrossEntropyLoss(reduction='none')

    print(f"Starting {optim_type} Training...")
    model.train()
    optimizer.zero_grad()

    step_count = 0
    acc_total_loss, acc_major_loss, acc_minor_loss = 0.0, 0.0, 0.0
    total_tokens, major_tokens, minor_tokens = 0, 0, 0

    # Fixed groups computed once from the tokenized training corpus.
    is_major = is_major.to(cfg.DEVICE)
    is_minor = is_minor.to(cfg.DEVICE)

    log_file_path = os.path.join("logs", f"training_loss_{optim_type.lower()}.txt")
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Starting {optim_type} Training Log...\n")
        log_file.write("========================\n")
        
        train_iter = iter(train_loader)
        batch_idx = 0
        
        while step_count < cfg.STEPS:
            try:
                inputs, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader) # Restart dataset (Epoch 2)
                inputs, labels = next(train_iter)

            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            
            logits = model(inputs)
            
            logits_flat = logits.view(-1, vocab_size)
            labels_flat = labels.view(-1)
            per_token_losses = criterion_none(logits_flat, labels_flat)

            total_loss = per_token_losses.mean()
            
            scaled_loss = total_loss / grad_acc_steps
            scaled_loss.backward()

            major_mask = is_major[labels_flat]
            minor_mask = is_minor[labels_flat]

            acc_total_loss += per_token_losses.sum().item()
            total_tokens += labels_flat.numel()
            
            if major_mask.any():
                acc_major_loss += per_token_losses[major_mask].sum().item()
                major_tokens += major_mask.sum().item()
                
            if minor_mask.any():
                acc_minor_loss += per_token_losses[minor_mask].sum().item()
                minor_tokens += minor_mask.sum().item()

            if (batch_idx + 1) % grad_acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 50 == 0:
                    avg_total = acc_total_loss / total_tokens if total_tokens > 0 else 0.0
                    avg_major = acc_major_loss / major_tokens if major_tokens > 0 else float("nan")
                    avg_minor = acc_minor_loss / minor_tokens if minor_tokens > 0 else float("nan")
                    
                    log_msg = (f"[{optim_type}] Step {step_count}/{cfg.STEPS} | "
                               f"Total Loss: {avg_total:.4f} | "
                               f"Major Loss: {avg_major:.4f} | "
                               f"Minor Loss: {avg_minor:.4f}")
                    print(log_msg)
                    
                    log_file.write(log_msg + "\n")
                    log_file.flush() 

                    acc_total_loss, acc_major_loss, acc_minor_loss = 0.0, 0.0, 0.0
                    total_tokens, major_tokens, minor_tokens = 0, 0, 0

                if step_count >= cfg.STEPS:
                    print(f"Training complete for {optim_type}. Logs saved to {log_file_path}")
                    break
            
            batch_idx += 1
                
    # Memory wipeout before running the next identical parallel sequential experiment
    del model, optimizer, train_loader, train_dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    os.makedirs("logs", exist_ok=True)
    
    train_file = download_wikitext103()
    
    # Train/Load BPE
    tokenizer = train_bpe_model(train_file)
    vocab_size = tokenizer.get_vocab_size()

    print("Counting training token frequencies...")
    counts, is_major, is_minor = build_frequency_groups(
        train_file, tokenizer, cfg.MAJOR_VOCAB_FRAC
    )
    np.savez(
        os.path.join("logs", "frequency_groups.npz"),
        counts=counts.numpy(), is_major=is_major.numpy(), is_minor=is_minor.numpy(),
    )
    tokenizer.save(os.path.join("logs", "tokenizer.json"))
    minority_mass = counts[is_minor].sum().item() / counts.sum().item()
    print(f"Minority: {is_minor.sum().item()} observed token types; "
          f"{minority_mass:.4%} of training token occurrences")

    print("Running AdamW Experiment...")
    run_experiment("AdamW", train_file, tokenizer, vocab_size, is_major, is_minor)
    
    print("\nRunning SGD Experiment...")
    run_experiment("SGD", train_file, tokenizer, vocab_size, is_major, is_minor)

    print("\nRunning Muon Experiment...")
    run_experiment("Muon", train_file, tokenizer, vocab_size, is_major, is_minor)
    
    print("\nAll experiments complete.")

if __name__ == "__main__":
    main()
