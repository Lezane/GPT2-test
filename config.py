import torch

# General Settings
INIT_STD = 0.02
DEPTH = 12
NUM_HEADS = 12
EMB_DIM = 768
SEED = 42  # Seed to ensure AdamW and SGD start from the exact same weights

# Training Settings
TARGET_LENGTH = 1024
STEPS = 1000
MAJOR_VOCAB_FRAC = 0.9  # Top 90% most common words

# AdamW Parameters (Matched to gpt2small_wt103.py)
ADAM_TRAIN_BATCH_SIZE = 8
ADAM_GRADIENT_ACC_STEPS = 16
ADAM_LEARNING_RATE = 1e-4
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.01

# SGD Parameters (Matched to gpt2small_wt103.py for 1GPU)
SGD_TRAIN_BATCH_SIZE = 32
SGD_GRADIENT_ACC_STEPS = 16
SGD_LEARNING_RATE = 0.01 # Exponent Fraction(-1, 1) mapping from standard grids = 10^-1
SGD_MOMENTUM = 0.9

# Hardware Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
