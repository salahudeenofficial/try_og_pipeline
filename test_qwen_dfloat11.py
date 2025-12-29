#!/usr/bin/env python3
"""
Qwen-Image-Edit-2509 DFloat11 Test Script

Uses DFloat11 LOSSLESS compression:
- 32% smaller model size
- 100% identical outputs (lossless!)
- Runs on 32GB GPU (or 24GB with CPU offload)

Model: DFloat11/Qwen-Image-Edit-2509-DF11
Base: Qwen/Qwen-Image-Edit-2509

Requirements:
    pip install -U dfloat11[cuda12]
    pip install git+https://github.com/huggingface/diffusers
"""

import argparse
import os
import sys
import time

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
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024 ** 3)
            print(f"\n📌 GPU {i}: {props.name}")
            print(f"   Total Memory: {total_memory:.2f} GB")
    else:
        print("❌ CUDA is not available!")
        sys.exit(1)
    
    print("=" * 60)
    return True


def load_pipeline_dfloat11(cpu_offload: bool = False, 
                           cpu_offload_blocks: int = 20,
                           no_pin_memory: bool = False):
    """
    Load the DFloat11 compressed pipeline.
    
    NOTE: DFloat11 is NOT compatible with LoRA (they both modify model layers).
    Use --no-lora mode (40 steps) for DFloat11 lossless compression.
    For fast 4-step inference, use the main branch without DFloat11.
    """
    from diffusers import QwenImageEditPlusPipeline
    from dfloat11 import DFloat11Model
    
    base_model_id = "Qwen/Qwen-Image-Edit-2509"
    df11_model_id = "DFloat11/Qwen-Image-Edit-2509-DF11"
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2509 with DFloat11 Compression")
    print("=" * 60)
    print(f"Base Model: {base_model_id}")
    print(f"DFloat11 Weights: {df11_model_id}")
    print(f"Compression: 32% smaller, 100% lossless!")
    print(f"Mode: 40 steps (DFloat11 is not compatible with LoRA)")
    print(f"CPU Offload: {cpu_offload}")
    print("-" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    # Load the base pipeline
    print("Loading base pipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
    )
    
    # Apply DFloat11 compression to transformer
    print("Applying DFloat11 compression to transformer...")
    DFloat11Model.from_pretrained(
        df11_model_id,
        bfloat16_model=pipeline.transformer,
        device="cpu",
        cpu_offload=cpu_offload,
        cpu_offload_blocks=cpu_offload_blocks,
        pin_memory=not no_pin_memory,
    )
    print("✅ DFloat11 compression applied!")
    
    # Enable CPU offload for memory efficiency
    pipeline.enable_model_cpu_offload()
    
    print(f"✅ Pipeline loaded in {time.time() - start_time:.2f} seconds")
    
    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"📊 GPU Memory: {allocated:.2f} GB / {total:.2f} GB")
    
    pipeline.set_progress_bar_config(disable=False)
    
    print("=" * 60)
    return pipeline


def run_inference(pipeline, 
                  images: list,
                  prompt: str,
                  output_path: str = "output_image.png",
                  num_inference_steps: int = 40,
                  true_cfg_scale: float = 4.0,
                  seed: int = 42):
    """Run inference with the pipeline."""
    print("\n" + "=" * 60)
    print("🎨 Running Virtual Try-On Inference (DFloat11)")
    print("=" * 60)
    print(f"Number of input images: {len(images)}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"True CFG scale: {true_cfg_scale}")
    print(f"Seed: {seed}")
    print("-" * 60)
    print(f"Prompt:\n{prompt[:200]}..." if len(prompt) > 200 else f"Prompt:\n{prompt}")
    print("-" * 60)
    
    inputs = {
        "image": images,
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": true_cfg_scale,
        "negative_prompt": " ",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
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
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"📊 Peak GPU Memory: {max_allocated:.2f} GB")
    
    print("=" * 60)
    return output_image


def main():
    parser = argparse.ArgumentParser(
        description="Qwen-Image-Edit-2509 DFloat11 Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This uses DFloat11 LOSSLESS compression - 32% smaller, 100% accuracy!

NOTE: DFloat11 is NOT compatible with Lightning LoRA.
For fast 4-step inference, use the main branch instead.

Install requirements:
    pip install -U dfloat11[cuda12]
    pip install git+https://github.com/huggingface/diffusers

Examples:
    # Standard (40 steps, lossless quality)
    python test_qwen_dfloat11.py --person person.jpg --cloth cloth.png
    
    # With CPU offload (24GB VRAM required)
    python test_qwen_dfloat11.py --person person.jpg --cloth cloth.png --cpu-offload
        """
    )
    parser.add_argument("--person", "-p", type=str, required=True,
                        help="Path to person image (with green mask)")
    parser.add_argument("--cloth", "-c", type=str, required=True,
                        help="Path to cloth/garment image")
    parser.add_argument("--output", "-o", type=str, default="vton_dfloat11_output.png",
                        help="Output path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (uses VTON prompt by default)")
    parser.add_argument("--steps", type=int, default=40,
                        help="Inference steps (default: 40 for base model)")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="True CFG scale (default: 4.0 for base model)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Enable CPU offloading (for 24GB GPUs)")
    parser.add_argument("--cpu-offload-blocks", type=int, default=20,
                        help="Number of blocks to offload to CPU")
    parser.add_argument("--no-pin-memory", action="store_true",
                        help="Disable memory pinning (most memory efficient)")
    
    args = parser.parse_args()
    
    # Default VTON prompt
    if args.prompt is None:
        args.prompt = """将图片 1 中的绿色遮罩区域仅用于判断服装属于上半身或下半身，不要将服装限制在遮罩范围内。

将图片 2 中的服装自然地穿戴到图片 1 中的人物身上，保持图片 2 中服装的完整形状、袖长和轮廓。无论图片 2 是单独的服装图还是人物穿着该服装的图，都应准确地转移服装，同时保留其原始面料质感、材质细节和颜色准确性。

确保图片 1 中人物的面部、头发和皮肤完全保持不变。光照与阴影应自然匹配图片 1 的环境，但服装的材质外观必须忠实于图片 2。

保持边缘平滑融合、阴影逼真，整体效果自然且不改变人物的身份特征。"""
    
    print("\n" + "👗" * 30)
    print("   QWEN VIRTUAL TRY-ON (DFloat11)")
    print("   Lossless Compression - 100% Quality!")
    print("   (40 steps - DFloat11 is not compatible with LoRA)")
    print("👗" * 30 + "\n")
    
    # Check CUDA
    check_cuda()
    
    # Check input files
    if not os.path.exists(args.person):
        print(f"❌ Person image not found: {args.person}")
        sys.exit(1)
    if not os.path.exists(args.cloth):
        print(f"❌ Cloth image not found: {args.cloth}")
        sys.exit(1)
    
    # Load images
    print("\n📷 Loading input images...")
    person_img = Image.open(args.person).convert("RGB")
    cloth_img = Image.open(args.cloth).convert("RGB")
    print(f"   Person image: {person_img.size}")
    print(f"   Cloth image: {cloth_img.size}")
    
    # Load pipeline with DFloat11 compression
    pipeline = load_pipeline_dfloat11(
        cpu_offload=args.cpu_offload,
        cpu_offload_blocks=args.cpu_offload_blocks,
        no_pin_memory=args.no_pin_memory,
    )
    
    # Run inference with both images
    output_image = run_inference(
        pipeline=pipeline,
        images=[person_img, cloth_img],
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        seed=args.seed,
    )
    
    print("\n" + "✅" * 30)
    print("   VIRTUAL TRY-ON COMPLETED!")
    print("   Using DFloat11 Lossless Compression (40-step)")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
