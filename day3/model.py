"""
Day 3: Minimal Transformer Implementation

A clean, minimal implementation of a GPT-like transformer model.
Implementation will be completed during Day 3 study session.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        # TODO: Implement multi-head attention
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        # TODO: Implement feed-forward network
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


class TransformerBlock(nn.Module):
    """Single transformer block"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        # TODO: Implement transformer block
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


class GPTModel(nn.Module):
    """Minimal GPT-like language model"""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        # TODO: Implement full model
        pass

    def forward(self, x):
        # TODO: Implement forward pass
        pass


if __name__ == "__main__":
    # Test model initialization
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 512
    max_seq_len = 256

    model = GPTModel(vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
