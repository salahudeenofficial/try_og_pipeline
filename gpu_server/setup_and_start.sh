#!/bin/bash
# GPU Server Setup and Start Script
# Installs requirements and starts the server on port 8000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "🚀 GPU Server Setup and Start"
echo "============================================================"
echo ""

# Step 1: Install parent requirements (ML frameworks)
echo "📦 Step 1: Installing parent requirements (PyTorch, etc.)..."
cd ..
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Parent requirements installed"
else
    echo "⚠️  Parent requirements.txt not found, skipping..."
fi

# Step 2: Install GPU server requirements
echo ""
echo "📦 Step 2: Installing GPU server requirements..."
cd "$SCRIPT_DIR"
pip install -r requirements.txt
echo "✅ GPU server requirements installed"

# Step 3: Verify configuration
echo ""
echo "📋 Step 3: Checking configuration..."
if [ ! -f "configs/config.yaml" ]; then
    echo "⚠️  Warning: configs/config.yaml not found!"
    exit 1
fi

echo "✅ Configuration file found"
echo ""
echo "📋 Current Callback Configuration:"
grep -A 5 "asset_service:" configs/config.yaml | head -6
echo ""
echo "📋 Server Port:"
grep "port:" configs/config.yaml | head -1
echo ""

# Step 4: Set environment variables
export CONFIG_PATH="${CONFIG_PATH:-configs/config.yaml}"
export PYTHONPATH="${SCRIPT_DIR}:${SCRIPT_DIR}/..:${PYTHONPATH}"

# Step 5: Start the server
echo "============================================================"
echo "🚀 Starting GPU Server on port 8000..."
echo "============================================================"
echo ""
echo "Server will be available at: http://0.0.0.0:8000"
echo "Health check: http://0.0.0.0:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
