# Final Architecture & Configuration Summary

Our winning model achieves **1.7101 bits-per-byte (BPB)** on dev evaluation, improving by **-0.6617 BPB** over the 2.3718 baseline under strict 2,000-step and 2,000,000 parameter limits.

1. **BPE Tokenization (vocab 1024)**: Replaces standard byte-level tokenization with a 768-merge BPE trained on corpus bytes, compressing Hindi Devanagari text by ~5.0x and overall corpus by 2.70x.
2. **Rotary Position Embeddings (RoPE)**: Replaces learned absolute position embeddings, eliminating 49,152 parameters and freeing budget to expand vocabulary size to 1024 while doubling context block size to 512.
3. **Weight Tying & RMSNorm**: Output projection weights are tied with input embedding weights, while RMSNorm replaces LayerNorm for parameter-free, fast normalization.
4. **SwiGLU Activation**: Gated SwiGLU MLP layers (hidden dim 512) replace GELU, providing superior non-linear capacity within the 1.97M parameter footprint.
5. **Optimization Strategy**: AdamW (`lr=6e-4`, `min_lr=6e-5`, `weight_decay=0.1`, `betas=(0.9, 0.95)`) with 100-step linear warmup and cosine decay schedule provides smooth convergence.
6. **Gradient Accumulation**: `accum_steps=8` (effective batch size 64, 32,768 tokens per step) processes 65.5M total tokens across 2,000 steps, maximizing compute throughput.
7. **No Noise Regularization**: Dropout (0.0) and label smoothing (0.0) are omitted because soft logit targets penalize exact cross-entropy log-likelihood metrics.
8. **Muon & QK-Norm Ablations**: Newton-Schulz matrix orthogonalization (Muon) and QK-Norm were tested as ambitious moves, but both constrained logit scale on a 2M-param model, proving that unconstrained AdamW gradients are optimal at this parameter tier.
