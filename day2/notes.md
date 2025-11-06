# Day 2: Notes from Karpathy's "Let's build GPT"

## Key Takeaways

### 1. Bigram Language Model
- Simplest possible language model
- Predicts next token based only on current token
- Baseline to compare against

### 2. Self-Attention
- Tokens "communicate" with each other
- Each token aggregates information from previous tokens
- Uses Query, Key, Value mechanism

### 3. Multi-Head Attention
- Multiple attention mechanisms run in parallel
- Each head can focus on different aspects
- Heads are concatenated and projected

### 4. Feed-Forward Network
- Applied to each position independently
- Two linear layers with ReLU activation
- Allows tokens to "think" on the aggregated information

### 5. Residual Connections & Layer Norm
- Residual connections: Add input to output of sublayer
- Helps with gradient flow during training
- Layer norm: Normalize across features

### 6. Position Embeddings
- Transformer has no built-in notion of position
- Need to inject position information
- Can use learned or sinusoidal embeddings

## Important Code Snippets to Understand

```python
# Self-attention head
# - Projects inputs to Q, K, V
# - Computes attention weights
# - Applies attention to values

# Multi-head attention
# - Concatenates multiple attention heads
# - Projects concatenated output

# Transformer block
# - Multi-head attention
# - Add & Norm
# - Feed-forward
# - Add & Norm
```

## Questions to Ask Yourself

- [ ] Why do we scale attention scores by sqrt(d_k)?
- [ ] What's the difference between encoder and decoder attention?
- [ ] Why do we need multiple attention heads?
- [ ] How do residual connections help training?
- [ ] What role does layer normalization play?

## Next Steps

- Implement each component from scratch
- Test each component individually
- Combine into full transformer
- Train on small dataset
