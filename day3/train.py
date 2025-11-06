"""
Day 3: Training Script

Simple training script for the minimal transformer model.
To be implemented during Day 3 study session.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    """Simple text dataset for character-level language modeling"""

    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        # TODO: Implement dataset initialization

    def __len__(self):
        # TODO: Implement length
        pass

    def __getitem__(self, idx):
        # TODO: Implement item retrieval
        pass


def train(model, dataloader, optimizer, criterion, device):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0

    # TODO: Implement training loop
    pass


def evaluate(model, dataloader, criterion, device):
    """Evaluation loop"""
    model.eval()
    total_loss = 0

    # TODO: Implement evaluation loop
    pass


def generate(model, start_text, max_len, temperature=1.0):
    """Generate text from the model"""
    # TODO: Implement text generation
    pass


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    seq_len = 128
    learning_rate = 3e-4
    num_epochs = 10

    # TODO: Load data, initialize model, train
    print("Training script ready for implementation")
