# Qwen-Image-Edit-2509 Pipeline

Run the **Qwen-Image-Edit-2509** model on Vast.ai GPU instances.

## 🖥️ Recommended Instance Configuration

| Parameter | Value |
|-----------|-------|
| **Base Image** | `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-ipv2` |
| **GPU** | NVIDIA L40S (45GB VRAM) or similar |
| **Disk Space** | 100GB+ (model is ~60GB) |
| **System RAM** | 64GB+ recommended |
| **Python** | 3.10 |
| **CUDA** | 12.4.1 |

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/salahudeenofficial/try_og_pipeline.git
cd try_og_pipeline
```

### 2. Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Install PyTorch with CUDA 12.4 support
- Install the latest diffusers from GitHub
- Install all other dependencies
- Verify the installation

### 3. Run a Test

**Basic test (creates a test image automatically):**
```bash
python test_qwen_edit.py
```

**With your own image:**
```bash
python test_qwen_edit.py \
    --input your_image.png \
    --prompt "Transform this into a watercolor painting"
```

**Multi-image editing:**
```bash
python test_qwen_edit.py \
    --input person1.png person2.png \
    --prompt "Both people are standing in a beautiful garden"
```

## 📋 Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Path to input image(s) | Creates test image |
| `--prompt`, `-p` | Edit prompt | "Transform this image into a watercolor painting style" |
| `--output`, `-o` | Output path | `output_image.png` |
| `--steps` | Inference steps | 40 |
| `--cfg` | True CFG scale | 4.0 |
| `--seed` | Random seed | 42 |
| `--skip-cuda-check` | Skip CUDA check | False |

## 🎨 Example Prompts

**Style Transfer:**
- "Transform this into an oil painting"
- "Make this look like a pencil sketch"
- "Convert to anime style"

**Scene Editing:**
- "Add a sunset in the background"
- "Place the person in a coffee shop"
- "Change the background to a beach"

**Person Editing:**
- "Make the person wear a red dress"
- "Change the hairstyle to curly"
- "Add sunglasses"

**Multi-Image (2-3 images):**
- "The two people are shaking hands in an office"
- "Place the product next to the person"
- "Combine these scenes into one panoramic view"

## 📊 Expected Performance

| Metric | Expected Value |
|--------|----------------|
| **Model Load Time** | 2-5 minutes (first run, downloads ~60GB) |
| **Inference Time** | 30-60 seconds (40 steps) |
| **VRAM Usage** | ~25-35GB |
| **Peak Memory** | ~40GB |

## 🔧 Troubleshooting

### Out of Memory (OOM)
If you get OOM errors, try:
```bash
# Reduce inference steps
python test_qwen_edit.py --steps 20

# Use smaller images (resize your input)
```

### Pipeline Not Found
Make sure you installed diffusers from git:
```bash
pip install git+https://github.com/huggingface/diffusers
```

### CUDA Errors
Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

## 📁 Project Structure

```
try_og_pipeline/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── setup.sh           # Setup script (run first)
├── test_qwen_edit.py  # Main test script
└── output_image.png   # Generated output (after running test)
```

## 📚 References

- [Qwen-Image-Edit-2509 on Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Vast.ai GPU Instances](https://vast.ai/)

## 📄 License

This code is provided for testing purposes. The Qwen-Image-Edit-2509 model has its own license - please refer to the [model card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) for details.
