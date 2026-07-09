import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import local modules
import config as cfg
from dataset import download_wikitext103, train_sentencepiece_model, SentencePieceWikiTextDataset
from model import SimpleGPT
from optimizer import configure_optimizers

def main():
    # Setup Logging Directory
    os.makedirs("logs", exist_ok=True)
    log_file_path = os.path.join("logs", "training_loss.txt")
    
    # Open log file to store the training loss
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Starting Training Log...\n")
        log_file.write("========================\n")
        
        train_file = download_wikitext103()

        sp = train_sentencepiece_model(train_file)
        vocab_size = sp.get_piece_size()

        train_dataset = SentencePieceWikiTextDataset(train_file, sp, cfg.TARGET_LENGTH)
        train_loader = DataLoader(train_dataset, batch_size=cfg.ADAM_TRAIN_BATCH_SIZE)

        print("Initializing Model...")
        model = SimpleGPT(
            vocab_size=vocab_size, emb_dim=cfg.EMB_DIM,
            num_heads=cfg.NUM_HEADS, depth=cfg.DEPTH
        ).to(cfg.DEVICE)

        optimizer = configure_optimizers(
            model, weight_decay=cfg.WEIGHT_DECAY, learning_rate=cfg.LEARNING_RATE,
            betas=(cfg.BETA1, cfg.BETA2), eps=cfg.EPS
        )

        criterion = nn.CrossEntropyLoss()

        print("Starting Training...")
        model.train()
        optimizer.zero_grad()

        step_count = 0
        accumulated_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(cfg.DEVICE), labels.to(cfg.DEVICE)

            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

            # Scale for accumulation
            loss = loss / cfg.ADAM_GRADIENT_ACC_STEPS
            loss.backward()
            accumulated_loss += loss.item()

            if (batch_idx + 1) % cfg.ADAM_GRADIENT_ACC_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

                if step_count % 50 == 0:
                    log_msg = f"Step {step_count}/{cfg.STEPS} | Loss: {accumulated_loss:.4f}"
                    print(log_msg)
                    # Write to file and flush immediately so progress is saved if interrupted
                    log_file.write(log_msg + "\n")
                    log_file.flush() 

                accumulated_loss = 0.0

                if step_count >= cfg.STEPS:
                    print(f"Training complete. Logs saved to {log_file_path}")
                    break

if __name__ == "__main__":
    main()