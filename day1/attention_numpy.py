"""
Day 1: Attention Mechanism Implementation using NumPy

This file contains a basic implementation of the attention mechanism
from scratch using only NumPy.
"""

import numpy as np


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(Q, K, V):
    """
    Scaled Dot-Product Attention

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)

    Returns:
        output: Attention output of shape (seq_len, d_v)
        attention_weights: Attention weights of shape (seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)

    # Apply softmax to get attention weights
    attention_weights = softmax(scores)

    # Compute weighted sum of values
    output = np.matmul(attention_weights, V)

    return output, attention_weights


if __name__ == "__main__":
    # Example usage
    seq_len = 4
    d_k = 8
    d_v = 8

    # Random initialization
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_v)

    # Compute attention
    output, weights = attention(Q, K, V)

    print("Query shape:", Q.shape)
    print("Key shape:", K.shape)
    print("Value shape:", V.shape)
    print("\nAttention output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    print("\nAttention weights:\n", weights)
