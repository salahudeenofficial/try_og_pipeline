#!/bin/bash
# ================================================================
# VTON GPU Server - One-Click Startup Script
# 
# This script sets up everything and starts the GPU server.
# Run this on a fresh Vast.ai instance with lightx2v docker image.
#
# Docker Image: lightx2v/lightx2v:25101501-cu124
# ================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "🚀 VTON GPU Server - One-Click Setup"
echo "========================================"
echo ""

# Step 1: Setup LightX2V (pinned to compatible version)
echo "📦 Step 1: Setting up LightX2V..."
if [ ! -d "LightX2V" ]; then
    echo "   Cloning LightX2V..."
    git clone https://github.com/ModelTC/LightX2V.git
fi

cd LightX2V
# IMPORTANT: Pin to commit before breaking changes
echo "   Checking out compatible version (7651b0f)..."
git fetch origin 2>/dev/null || true
git checkout 7651b0f 2>/dev/null || git checkout -f 7651b0f
pip install -q -e . 2>/dev/null || pip install -e .
cd ..
echo "   ✅ LightX2V ready"

# Step 2: Download models
echo ""
echo "📥 Step 2: Downloading models..."

mkdir -p models

# Download base model if not present
if [ ! -d "models/Qwen-Image-Edit-2511" ] || [ -z "$(ls -A models/Qwen-Image-Edit-2511 2>/dev/null)" ]; then
    echo "   Downloading Qwen-Image-Edit-2511 base model..."
    huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511
else
    echo "   ✅ Base model already downloaded"
fi

# Download Lightning/FP8 weights if not present
if [ ! -d "models/Qwen-Image-Edit-2511-Lightning" ] || [ -z "$(ls -A models/Qwen-Image-Edit-2511-Lightning 2>/dev/null)" ]; then
    echo "   Downloading Lightning LoRA and FP8 weights..."
    huggingface-cli download lightx2v/Qwen-Image-Edit-2511-Lightning --local-dir models/Qwen-Image-Edit-2511-Lightning
else
    echo "   ✅ Lightning weights already downloaded"
fi

# Verify FP8 weights exist
FP8_FILE="models/Qwen-Image-Edit-2511-Lightning/qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors"
if [ -f "$FP8_FILE" ]; then
    echo "   ✅ FP8 weights found"
else
    echo "   ⚠️  FP8 weights not found at expected path!"
    echo "      Looking for: $FP8_FILE"
    echo "      Available files:"
    ls -la models/Qwen-Image-Edit-2511-Lightning/ 2>/dev/null || echo "   (folder not found)"
fi

# Step 3: Install GPU server dependencies
echo ""
echo "📦 Step 3: Installing GPU server dependencies..."
pip install -q fastapi uvicorn httpx pyyaml pydantic pillow 2>/dev/null || \
pip install fastapi uvicorn httpx pyyaml pydantic pillow
echo "   ✅ Dependencies installed"

# Step 4: Start the server
echo ""
echo "========================================"
echo "🚀 Starting GPU Server"
echo "========================================"
echo ""
echo "   Port: 8000"
echo "   Frontend: Connect to http://<your-ip>:8000"
echo ""
echo "   Test endpoints:"
echo "     curl http://localhost:8000/health"
echo "     curl http://localhost:8000/test"
echo ""
echo "   Press Ctrl+C to stop"
echo ""
echo "========================================"

# Start the server
cd gpu_server
export CONFIG_PATH="configs/config.yaml"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/gpu_server:${PYTHONPATH}"

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
