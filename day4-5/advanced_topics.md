# Day 4-5: Advanced Topics and Optimizations

## Topics to Explore

### 1. Understanding the Original Paper
- **Attention Is All You Need** (Vaswani et al., 2017)
- Focus on:
  - Why positional encoding uses sine/cosine functions
  - The role of layer normalization
  - Multi-head attention mathematical formulation
  - Encoder-decoder vs decoder-only architectures

### 2. Optimizations

#### Model Architecture
- [ ] Pre-layer normalization vs post-layer normalization
- [ ] Different activation functions (GELU, SwiGLU)
- [ ] Rotary Position Embeddings (RoPE)
- [ ] Grouped Query Attention (GQA)

#### Training Techniques
- [ ] Learning rate scheduling (cosine decay, warmup)
- [ ] Gradient clipping
- [ ] Mixed precision training
- [ ] Gradient accumulation

#### Inference Optimizations
- [ ] KV caching for autoregressive generation
- [ ] Beam search vs sampling strategies
- [ ] Temperature and top-k/top-p sampling

### 3. Scaling Considerations

#### Making Models Larger
- Parameter count vs compute tradeoffs
- Depth vs width
- MoE (Mixture of Experts)

#### Making Training Efficient
- Gradient checkpointing
- Distributed training basics
- Flash Attention

### 4. Practical Considerations

#### Debugging Tips
```python
# Always print shapes during development
print(f"Input shape: {x.shape}")
print(f"After attention: {attn_out.shape}")
print(f"After FFN: {ffn_out.shape}")

# Check for NaN/Inf
assert not torch.isnan(x).any(), "NaN detected!"
assert not torch.isinf(x).any(), "Inf detected!"

# Monitor gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad norm: {param.grad.norm()}")
```

#### Common Issues and Solutions
- **Exploding gradients**: Use gradient clipping
- **Vanishing gradients**: Check residual connections
- **Slow convergence**: Adjust learning rate, check warmup
- **Out of memory**: Reduce batch size, use gradient accumulation

### 5. Reading Research Papers

#### Essential Papers
1. **Attention Is All You Need** (2017)
   - Original transformer paper
   - Foundation for everything else

2. **GPT-1: Improving Language Understanding** (2018)
   - Decoder-only architecture
   - Pre-training + fine-tuning paradigm

3. **GPT-2: Language Models are Unsupervised Multitask Learners** (2019)
   - Scaling up
   - Zero-shot learning

4. **GPT-3: Language Models are Few-Shot Learners** (2020)
   - In-context learning
   - Scaling laws

#### How to Read Papers
1. Read abstract and introduction first
2. Look at figures and tables
3. Skim methodology
4. Deep dive into sections relevant to your interest
5. Try to implement key ideas

### 6. Extensions and Variations

#### Encoder-Decoder Models
- BERT (bidirectional)
- T5 (text-to-text)
- BART (denoising autoencoder)

#### Decoder-Only Models
- GPT series
- LLaMA
- Mistral

#### Specialized Architectures
- Vision Transformers (ViT)
- Audio transformers (Whisper)
- Multimodal transformers (CLIP)

## Exercises

### Exercise 1: Implement Pre-LN vs Post-LN
Compare training dynamics of pre-layer norm and post-layer norm

### Exercise 2: Add KV Caching
Implement efficient generation with KV caching

### Exercise 3: Visualize Attention
Create attention weight visualizations to understand what the model learns

### Exercise 4: Experiment with Architectures
- Try different numbers of heads
- Vary model depth and width
- Compare performance and training time

## Resources

### Implementation References
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [minGPT](https://github.com/karpathy/minGPT)
- [llama2.c](https://github.com/karpathy/llama2.c)

### Deep Dives
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
- [Transformer Circuits Thread](https://transformer-circuits.pub/)

### Advanced Topics
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Rotary Position Embeddings](https://arxiv.org/abs/2104.09864)
- [Scaling Laws](https://arxiv.org/abs/2001.08361)

## Next Steps After This Study

- Implement BERT (encoder-only)
- Try multi-task learning
- Explore instruction fine-tuning
- Study RLHF (Reinforcement Learning from Human Feedback)
- Look into efficient inference techniques

---

**Remember**: Understanding comes from implementation. Don't just read about these topics - implement them!
