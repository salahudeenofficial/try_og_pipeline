#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 Test Script with 4-Step Lightning LoRA

This script tests the Qwen-Image-Edit-2511 model with the 4-step Lightning LoRA
for ~10x faster inference on a Vast.ai instance.

Base Image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-ipv2

Requirements:
- L40S GPU with 45GB VRAM
- CUDA 12.4+
- Python 3.10

Models:
- Base: Qwen/Qwen-Image-Edit-2511
- LoRA: lightx2v/Qwen-Image-Edit-2511-Lightning (4-step distilled)

Usage:
    python test_qwen_edit.py
    python test_qwen_edit.py --input path/to/image.png --prompt "Your edit prompt"
    python test_qwen_edit.py --no-lora  # Use base model (40 steps)
"""

import argparse
import math
import os
import sys
import time
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image


def check_cuda():
    """Check CUDA availability and print GPU info."""
    print("=" * 60)
    print("🔍 CUDA Environment Check")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 ** 3)
            print(f"\n📌 GPU {i}: {props.name}")
            print(f"   Total Memory: {total_memory:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multi-Processor Count: {props.multi_processor_count}")
    else:
        print("❌ CUDA is not available!")
        sys.exit(1)
    
    print("=" * 60)
    return True


def download_sample_image():
    """Download a sample image from Qwen's demo repository."""
    url = "https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen-Image/edit2511/edit2511input.png"
    print(f"📥 Downloading sample image from: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # Save locally
        local_path = "sample_input.png"
        img.save(local_path)
        print(f"✅ Sample image saved to: {local_path}")
        return local_path
    except Exception as e:
        print(f"⚠️ Could not download sample image: {e}")
        return None


def create_test_image(output_path: str = "test_input.png", size: tuple = (512, 512)):
    """Create a simple test image if no input is provided."""
    from PIL import ImageDraw
    
    # Create a gradient background with some shapes
    img = Image.new('RGB', size, color=(70, 130, 180))  # Steel blue
    draw = ImageDraw.Draw(img)
    
    # Draw some shapes
    draw.ellipse([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], 
                 fill=(255, 200, 100), outline=(255, 150, 50), width=3)
    
    # Add text
    try:
        draw.text((size[0]//2, size[1]//2), "Test Image", 
                  fill=(255, 255, 255), anchor="mm")
    except:
        draw.text((size[0]//3, size[1]//2), "Test Image", fill=(255, 255, 255))
    
    img.save(output_path)
    print(f"✅ Created test image: {output_path}")
    return output_path


def download_lora_weights(lora_dir: str = "./lora_weights"):
    """Download the 4-step Lightning LoRA weights from Hugging Face."""
    from huggingface_hub import hf_hub_download
    
    os.makedirs(lora_dir, exist_ok=True)
    
    lora_filename = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
    lora_path = os.path.join(lora_dir, lora_filename)
    
    if os.path.exists(lora_path):
        print(f"✅ LoRA weights already exist: {lora_path}")
        return lora_path
    
    print("\n" + "=" * 60)
    print("📥 Downloading 4-Step Lightning LoRA Weights")
    print("=" * 60)
    print("Repository: lightx2v/Qwen-Image-Edit-2511-Lightning")
    print(f"File: {lora_filename}")
    print("This may take a few minutes...")
    print("-" * 60)
    
    try:
        downloaded_path = hf_hub_download(
            repo_id="lightx2v/Qwen-Image-Edit-2511-Lightning",
            filename=lora_filename,
            local_dir=lora_dir,
            local_dir_use_symlinks=False,
        )
        print(f"✅ LoRA weights downloaded to: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"❌ Failed to download LoRA weights: {e}")
        print("You can manually download from: https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning")
        return None


def load_pipeline(model_id: str = "Qwen/Qwen-Image-Edit-2511", 
                  lora_path: str = None,
                  device: str = "cuda",
                  dtype: torch.dtype = torch.bfloat16):
    """Load the Qwen-Image-Edit-2511 pipeline with optional Lightning LoRA."""
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.models import QwenImageTransformer2DModel
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2511 Pipeline")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"LoRA: {'✅ 4-Step Lightning' if lora_path else '❌ Base Model (40 steps)'}")
    print("This may take several minutes on first run...")
    print("-" * 60)
    
    # Clear any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    if lora_path is not None and os.path.exists(lora_path):
        # Scheduler config for distilled LoRA (shift=3)
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # shift=3 used in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # shift=3 used in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set to None for distilled model
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        
        # Load pipeline directly to GPU with device_map
        print("Loading full pipeline with custom scheduler (direct to GPU)...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=dtype,
            device_map="balanced",  # Automatically distribute across devices
        )
        
        print(f"Loading LoRA weights from: {lora_path}")
        pipeline.load_lora_weights(lora_path)
        print("✅ LoRA weights loaded successfully!")
        
    else:
        # Load base model without LoRA - direct to GPU
        print("Loading base pipeline (no LoRA, direct to GPU)...")
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="balanced",
        )
    
    print(f"✅ Pipeline loaded in {time.time() - start_time:.2f} seconds")
    
    # Enable CPU offload for memory efficiency if needed
    try:
        # Check memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"📊 GPU Memory - Allocated: {allocated:.2f} GB / {total:.2f} GB")
            
            # If we're using more than 90% of memory, enable offloading
            if allocated / total > 0.9:
                print("⚠️ Memory usage high, enabling sequential CPU offload...")
                pipeline.enable_sequential_cpu_offload()
    except Exception as e:
        print(f"Note: Could not check/enable CPU offload: {e}")
    
    pipeline.set_progress_bar_config(disable=False)
    
    print("=" * 60)
    return pipeline


def run_inference(pipeline, 
                  images: list,
                  prompt: str,
                  output_path: str = "output_image.png",
                  num_inference_steps: int = 4,
                  guidance_scale: float = 1.0,
                  true_cfg_scale: float = 1.0,
                  seed: int = 42,
                  use_lora: bool = True):
    """Run inference with the pipeline."""
    print("\n" + "=" * 60)
    print("🎨 Running Image Edit Inference")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Number of input images: {len(images)}")
    print(f"Inference steps: {num_inference_steps} {'(Lightning LoRA)' if use_lora else '(Base Model)'}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    # For multi-image input, pass as list; for single image, can pass directly
    image_input = images if len(images) > 1 else images
    
    inputs = {
        "image": image_input,
        "prompt": prompt,
        "generator": torch.Generator(device="cuda").manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
    }
    
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(**inputs)
    
    inference_time = time.time() - start_time
    print(f"\n✅ Inference completed in {inference_time:.2f} seconds")
    print(f"⚡ Speed: {inference_time / num_inference_steps:.2f} seconds per step")
    
    # Save output
    output_image = output.images[0]
    output_image.save(output_path)
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")
    
    # Print final memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 GPU Memory - Current: {allocated:.2f} GB, Peak: {max_allocated:.2f} GB")
    
    print("=" * 60)
    return output_image


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen-Image-Edit-2511 with 4-Step Lightning LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with sample image
  python test_qwen_edit.py

  # With your own image
  python test_qwen_edit.py --input your_image.png --prompt "Add sunset lighting"

  # Multi-image editing
  python test_qwen_edit.py --input person1.png person2.png --prompt "Both people in a garden"

  # Use base model (slower but may be higher quality)
  python test_qwen_edit.py --no-lora --steps 40 --cfg 4.0
        """
    )
    parser.add_argument("--input", "-i", type=str, nargs="+", default=None,
                        help="Path to input image(s). If not provided, downloads a sample image.")
    parser.add_argument("--prompt", "-p", type=str, 
                        default="Transform this into a beautiful oil painting with dramatic lighting",
                        help="Edit prompt for the image")
    parser.add_argument("--output", "-o", type=str, default="output_image.png",
                        help="Path for output image")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of inference steps (default: 4 with LoRA, 40 without)")
    parser.add_argument("--cfg", type=float, default=None,
                        help="True CFG scale (default: 1.0 with LoRA, 4.0 without)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-lora", action="store_true",
                        help="Use base model without Lightning LoRA (slower)")
    parser.add_argument("--skip-cuda-check", action="store_true",
                        help="Skip CUDA environment check")
    parser.add_argument("--lora-dir", type=str, default="./lora_weights",
                        help="Directory for LoRA weights")
    
    args = parser.parse_args()
    
    # Determine if using LoRA
    use_lora = not args.no_lora
    
    # Set defaults based on LoRA usage
    if args.steps is None:
        num_steps = 4 if use_lora else 40
    else:
        num_steps = args.steps
    
    if args.cfg is None:
        cfg_scale = 1.0 if use_lora else 4.0
    else:
        cfg_scale = args.cfg
    
    print("\n" + "🎨" * 30)
    print("   QWEN-IMAGE-EDIT-2511 TEST SCRIPT")
    print("   With 4-Step Lightning LoRA ⚡" if use_lora else "   Base Model (40 steps)")
    print("🎨" * 30 + "\n")
    
    # Check CUDA
    if not args.skip_cuda_check:
        check_cuda()
    
    # Download LoRA weights if needed
    lora_path = None
    if use_lora:
        lora_path = download_lora_weights(args.lora_dir)
        if lora_path is None:
            print("⚠️ Could not get LoRA weights, falling back to base model")
            use_lora = False
            num_steps = 40
            cfg_scale = 4.0
    
    # Prepare input images
    if args.input is None:
        print("\n📷 No input image provided, downloading sample image...")
        sample_path = download_sample_image()
        if sample_path is None:
            print("Creating a simple test image instead...")
            sample_path = create_test_image()
        input_paths = [sample_path]
    else:
        input_paths = args.input
        for path in input_paths:
            if not os.path.exists(path):
                print(f"❌ Input image not found: {path}")
                sys.exit(1)
    
    # Load images
    images = [Image.open(p).convert("RGB") for p in input_paths]
    print(f"\n📷 Loaded {len(images)} image(s)")
    for i, img in enumerate(images):
        print(f"   Image {i+1}: {img.size} from {input_paths[i]}")
    
    # Load pipeline
    pipeline = load_pipeline(
        model_id="Qwen/Qwen-Image-Edit-2511",
        lora_path=lora_path if use_lora else None,
    )
    
    # Run inference
    output_image = run_inference(
        pipeline=pipeline,
        images=images,
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=num_steps,
        true_cfg_scale=cfg_scale,
        seed=args.seed,
        use_lora=use_lora,
    )
    
    print("\n" + "✅" * 30)
    print("   TEST COMPLETED SUCCESSFULLY!")
    print("✅" * 30 + "\n")
    
    # Print summary
    print("📋 Summary:")
    print(f"   Model: Qwen-Image-Edit-2511")
    print(f"   LoRA: {'4-Step Lightning ⚡' if use_lora else 'None (Base Model)'}")
    print(f"   Steps: {num_steps}")
    print(f"   CFG Scale: {cfg_scale}")
    print(f"   Output: {os.path.abspath(args.output)}")
    print("")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
