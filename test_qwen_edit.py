#!/usr/bin/env python3
"""
Qwen-Image-Edit-2509 Test Script

This script tests the Qwen-Image-Edit-2509 model on a Vast.ai instance.
Base Image: vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310-ipv2

Requirements:
- L40S GPU with 45GB VRAM
- CUDA 12.4.1
- Python 3.10

Usage:
    python test_qwen_edit.py
    python test_qwen_edit.py --input path/to/image.png --prompt "Your edit prompt"
"""

import os
import sys
import argparse
import time
from pathlib import Path

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


def create_test_image(output_path: str = "test_input.png", size: tuple = (512, 512)):
    """Create a simple test image if no input is provided."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a gradient background
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


def load_pipeline(model_id: str = "Qwen/Qwen-Image-Edit-2509", 
                  device: str = "cuda",
                  dtype: torch.dtype = torch.bfloat16):
    """Load the Qwen-Image-Edit-2509 pipeline."""
    from diffusers import QwenImageEditPlusPipeline
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2509 Pipeline")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print("This may take several minutes on first run...")
    print("-" * 60)
    
    start_time = time.time()
    
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    
    print(f"✅ Pipeline loaded in {time.time() - start_time:.2f} seconds")
    
    # Move to GPU
    print("Moving pipeline to GPU...")
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"📊 GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    print("=" * 60)
    return pipeline


def run_inference(pipeline, 
                  images: list,
                  prompt: str,
                  output_path: str = "output_image.png",
                  num_inference_steps: int = 40,
                  guidance_scale: float = 1.0,
                  true_cfg_scale: float = 4.0,
                  seed: int = 42):
    """Run inference with the pipeline."""
    print("\n" + "=" * 60)
    print("🎨 Running Image Edit Inference")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Number of input images: {len(images)}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print("-" * 60)
    
    inputs = {
        "image": images if len(images) > 1 else images[0],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }
    
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(**inputs)
    
    inference_time = time.time() - start_time
    print(f"\n✅ Inference completed in {inference_time:.2f} seconds")
    
    # Save output
    output_image = output.images[0]
    output_image.save(output_path)
    print(f"💾 Output saved to: {os.path.abspath(output_path)}")
    
    # Print final memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 Peak GPU Memory: {max_allocated:.2f} GB")
    
    print("=" * 60)
    return output_image


def main():
    parser = argparse.ArgumentParser(description="Test Qwen-Image-Edit-2509 Model")
    parser.add_argument("--input", "-i", type=str, nargs="+", default=None,
                        help="Path to input image(s). If not provided, creates a test image.")
    parser.add_argument("--prompt", "-p", type=str, 
                        default="Transform this image into a watercolor painting style",
                        help="Edit prompt for the image")
    parser.add_argument("--output", "-o", type=str, default="output_image.png",
                        help="Path for output image")
    parser.add_argument("--steps", type=int, default=40,
                        help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="True CFG scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--skip-cuda-check", action="store_true",
                        help="Skip CUDA environment check")
    
    args = parser.parse_args()
    
    print("\n" + "🎨" * 30)
    print("   QWEN-IMAGE-EDIT-2509 TEST SCRIPT")
    print("🎨" * 30 + "\n")
    
    # Check CUDA
    if not args.skip_cuda_check:
        check_cuda()
    
    # Prepare input images
    if args.input is None:
        print("\n📷 No input image provided, creating test image...")
        input_paths = [create_test_image()]
    else:
        input_paths = args.input
        for path in input_paths:
            if not os.path.exists(path):
                print(f"❌ Input image not found: {path}")
                sys.exit(1)
    
    # Load images
    images = [Image.open(p).convert("RGB") for p in input_paths]
    print(f"📷 Loaded {len(images)} image(s)")
    for i, img in enumerate(images):
        print(f"   Image {i+1}: {img.size}")
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Run inference
    output_image = run_inference(
        pipeline=pipeline,
        images=images,
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        seed=args.seed,
    )
    
    print("\n" + "✅" * 30)
    print("   TEST COMPLETED SUCCESSFULLY!")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
