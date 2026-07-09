import torch

# General Settings
INIT_STD = 0.02
DEPTH = 12
NUM_HEADS = 12
EMB_DIM = 768

# Training Settings
TARGET_LENGTH = 1024
STEPS = 5000
ADAM_TRAIN_BATCH_SIZE = 8
ADAM_GRADIENT_ACC_STEPS = 16

# AdamW Parameters
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.95
EPS = 1e-8
WEIGHT_DECAY = 0.01

# Hardware Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")