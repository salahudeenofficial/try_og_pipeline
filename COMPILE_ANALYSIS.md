# LightX2V Qwen Image Model - torch.compile Analysis

## Overview

This document analyzes the LightX2V Qwen Image model structure to identify components
that can be compiled with `torch.compile()` for performance optimization.

## Model Architecture

```
QwenImageTransformerModel
├── pre_infer (QwenImagePreInfer)       # Preprocessing
├── transformer_infer (QwenImageTransformerInfer)  # Main transformer blocks
├── post_infer (QwenImagePostInfer)     # Postprocessing
├── pre_weight (QwenImagePreWeights)    # Weights for pre_infer
├── transformer_weights (QwenImageTransformerWeights)  # Transformer weights
└── post_weight (QwenImagePostWeights)  # Weights for post_infer
```

## Key Inference Flow

1. **`model.infer(inputs)`** - Main entry point
   - Calls `_infer_cond_uncond()` for the actual inference
   
2. **`_infer_cond_uncond(latents_input, prompt_embeds)`**:
   ```python
   pre_infer_out = self.pre_infer.infer(...)
   hidden_states = self.transformer_infer.infer(...)  # <-- HOTSPOT
   noise_pred = self.post_infer.infer(...)
   ```

3. **`transformer_infer.infer()`** contains the expensive operations:
   - Multiple transformer blocks (`num_layers=60`)
   - Attention computations (QKV projections, cross-attention)
   - RoPE embeddings
   - Modulation functions

## Compilable Components

### 1. **`transformer_infer.infer`** (HIGH IMPACT)
- Location: `QwenImageTransformerInfer.infer()`
- This is the main computational hotspot
- Contains 60 transformer blocks
- **Compilation Strategy**: Compile this method directly

### 2. **`transformer_infer.infer_calculating`** (HIGH IMPACT)  
- Internal method that runs the actual block computations
- Called from `infer()` via `self.infer_func`

### 3. **Individual Block Operations** (MEDIUM IMPACT)
- `infer_modulate()` - Modulation operations
- `infer_img_qkv()` - Image QKV projections
- `infer_txt_qkv()` - Text QKV projections
- `infer_cross_attn()` - Cross attention

### 4. **Pre/Post Processing** (LOW IMPACT)
- `pre_infer.infer()` - Minimal computation
- `post_infer.infer()` - Minimal computation

## Recommended Compilation Strategy

### Option A: Compile the entire transformer_infer (Easiest)
```python
# After model is loaded
pipe.runner.model.transformer_infer.infer = torch.compile(
    pipe.runner.model.transformer_infer.infer,
    mode="reduce-overhead",
    fullgraph=False
)
```

### Option B: Compile _infer_cond_uncond (Full path)
```python
pipe.runner.model._infer_cond_uncond = torch.compile(
    pipe.runner.model._infer_cond_uncond,
    mode="reduce-overhead"
)
```

### Option C: Compile infer_calculating (Core computation)
```python
pipe.runner.model.transformer_infer.infer_calculating = torch.compile(
    pipe.runner.model.transformer_infer.infer_calculating,
    mode="max-autotune"  # For maximum optimization
)
```

## Potential Issues

1. **Dynamic Shapes**: The model uses dynamic sequence lengths which may require
   `fullgraph=False` or `dynamic=True`

2. **Custom Operations**: Uses triton kernels (`fuse_scale_shift_kernel`) which
   may not be compatible with torch.compile

3. **Attention Backend**: Flash Attention may have compilation issues

4. **First Run Overhead**: Compilation takes 30-120 seconds on first run

## Implementation Plan

1. Create wrapper that applies `torch.compile` after model loading
2. Test with `fullgraph=False` first for compatibility
3. Measure performance improvement
4. Iterate to find optimal compilation settings

## Expected Performance Improvement

- **10-30% faster inference** after warmup
- Trade-off: First inference takes 30-120s for compilation
- Compilation is cached for subsequent runs

## Next Steps

1. Implement Option A (simplest approach)
2. Test with 480p resolution
3. Measure warmup time and steady-state performance
4. If issues, try Option C with specific components
