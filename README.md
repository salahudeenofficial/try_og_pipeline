# LightX2V Qwen-Image-Edit-2511 with FP8 Quantization ⚡

High-performance Virtual Try-On using **LightX2V** framework with **Qwen-Image-Edit-2511** model.

## 🚀 Performance

| Mode | Steps | Time (H100) | VRAM | Speedup |
|------|-------|-------------|------|---------|
| **FP8 + Distill** | 4 | ~3-4s | ~15-20GB | **42x** |
| **BF16 + LoRA** | 4 | ~4-6s | ~25-30GB | 10x |
| **Base Model** | 40 | ~40-60s | ~35GB | 1x |

## 📦 Quick Start

### 1. Setup

```bash
# Clone the repo
git clone https://github.com/salahudeenofficial/try_og_pipeline.git
cd try_og_pipeline
git checkout lightx2v

# Run setup (installs LightX2V, downloads models)
chmod +x setup_lightx2v.sh
./setup_lightx2v.sh
```

### 2. Run Virtual Try-On

```bash
# FP8 mode (fastest, recommended for production)
python test_lightx2v_vton.py --mode fp8 --person person.jpg --cloth cloth.png

# LoRA mode (BF16 + 4-step distillation)
python test_lightx2v_vton.py --mode lora --person person.jpg --cloth cloth.png

# With CPU offloading (for low VRAM GPUs like RTX 3090)
python test_lightx2v_vton.py --mode fp8 --offload
```

## 🔧 Models

### Required Downloads

All models are downloaded automatically by `setup_lightx2v.sh`. Manual download:

```bash
# Base model (~50GB)
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511

# Lightning models (LoRA + FP8)
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/Qwen-Image-Edit-2511-Lightning
```

### Model Files

| File | Size | Description |
|------|------|-------------|
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | ~850MB | 4-step LoRA for diffusers |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors` | ~1.7GB | 4-step LoRA (full precision) |
| `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning.safetensors` | ~10GB | FP8 base + distillation |

## 🎯 Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | `fp8`, `lora`, or `base` | `lora` |
| `--person` | Person image path | `person.jpg` |
| `--cloth` | Cloth image path | `cloth.png` |
| `--output` | Output path | `outputs/vton_lightx2v_result.png` |
| `--offload` | Enable CPU offloading | False |
| `--seed` | Random seed | 42 |
| `--prompt-lang` | `cn` or `en` | `cn` |

## 📊 Memory Requirements

| Mode | Min VRAM | Recommended | Notes |
|------|----------|-------------|-------|
| **FP8** | 16GB | 24GB | Best for RTX 4090, A100 |
| **FP8 + Offload** | 8GB | 16GB | For RTX 3080/3090 |
| **LoRA** | 24GB | 32GB | Standard mode |
| **Base** | 32GB | 48GB | Full precision, 40 steps |

## 🔗 References

- [LightX2V GitHub](https://github.com/ModelTC/LightX2V)
- [Qwen-Image-Lightning GitHub](https://github.com/ModelTC/Qwen-Image-Lightning)
- [Qwen-Image-Edit-2511 Model](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [Lightning Models](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)

## 📁 Directory Structure

```
try_og_pipeline/
├── setup_lightx2v.sh          # Setup script
├── test_lightx2v_vton.py      # Main VTON test script
├── person.jpg                 # Test person image
├── cloth.png                  # Test cloth image
├── models/
│   ├── Qwen-Image-Edit-2511/              # Base model
│   └── Qwen-Image-Edit-2511-Lightning/    # LoRA + FP8
├── outputs/                   # Generated images
└── LightX2V/                  # LightX2V framework
```

## 📄 License

- Qwen-Image-Edit-2511: Apache 2.0
- LightX2V: Apache 2.0
- Lightning LoRA: Same as base model
