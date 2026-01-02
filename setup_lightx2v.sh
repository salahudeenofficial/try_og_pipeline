#!/bin/bash
# ================================================================
# LightX2V Setup Script for Qwen-Image-Edit-2511
# 
# This script sets up the LightX2V framework for high-performance
# inference with FP8 quantization on Qwen-Image-Edit-2511.
# ================================================================

set -e  # Exit on error

echo "========================================"
echo "🚀 LightX2V Setup for Qwen-Image-Edit-2511"
echo "========================================"
echo ""

# Print system info
echo "📋 System Information:"
echo "  Python: $(python3 --version)"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install PyTorch 2.6.0 with CUDA 12.4 support
echo ""
echo "🔧 Installing PyTorch 2.6.0 with CUDA 12.4 support..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Clone and install LightX2V
echo ""
echo "🔧 Installing LightX2V..."
if [ ! -d "LightX2V" ]; then
    git clone https://github.com/ModelTC/LightX2V.git
fi

cd LightX2V
pip install -v -e .
cd ..

# Install additional dependencies
echo ""
echo "🔧 Installing additional dependencies..."
pip install transformers>=4.51.3
pip install accelerate>=0.30.0
pip install safetensors>=0.4.0
pip install Pillow>=10.0.0
pip install requests>=2.31.0
pip install tqdm>=4.66.0
pip install einops>=0.7.0

# Install flash-attn for optimal performance
echo ""
echo "🔧 Installing Flash Attention (optional but recommended)..."
pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed, will use fallback"

# Create models directory
echo ""
echo "📁 Creating directories..."
mkdir -p models
mkdir -p outputs

# Download models from HuggingFace
echo ""
echo "📥 Downloading models..."
pip install "huggingface_hub[cli]"

# Download base model (if not already present)
if [ ! -d "models/Qwen-Image-Edit-2511" ]; then
    echo "Downloading Qwen-Image-Edit-2511 base model..."
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
fi

# Download Lightning models (LoRA + FP8)
echo "Downloading Lightning LoRA and FP8 models..."
huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/Qwen-Image-Edit-2511-Lightning

echo ""
echo "========================================"
echo "✅ LightX2V Setup Complete!"
echo "========================================"
echo ""
echo "📁 Directory structure:"
echo "  models/"
echo "  ├── Qwen-Image-Edit-2511/           # Base model"
echo "  ├── Qwen-Image-Edit-2511-Lightning/ # LoRA + FP8 weights"
echo "  outputs/                             # Generated images"
echo ""
echo "📝 Available scripts:"
echo ""
echo "  1. Run VTON with BF16 + 4-step LoRA:"
echo "     python test_lightx2v_vton.py --mode lora"
echo ""
echo "  2. Run VTON with FP8 + 4-step distillation:"
echo "     python test_lightx2v_vton.py --mode fp8"
echo ""
echo "  3. Run with CPU offloading (low VRAM):"
echo "     python test_lightx2v_vton.py --mode fp8 --offload"
echo ""
