#!/usr/bin/env python3
"""
Qwen-Image-Edit-2511 SDNQ UINT4 Test Script

Uses SDNQ (Squeezed Data Neural Quantization) for 4-bit quantization.
This is DIRECTLY compatible with diffusers!

Model: Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32
- 4-bit quantization with SVD rank 32
- Expected VRAM: ~12-15GB
- Compatible with QwenImageEditPlusPipeline

Requirements:
- pip install sdnq
"""

import argparse
import math
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


def load_pipeline_sdnq(device: str = "cuda"):
    """Load the SDNQ 4-bit quantized pipeline."""
    import diffusers
    from diffusers import FlowMatchEulerDiscreteScheduler
    
    # Import SDNQ - this registers it into diffusers
    from sdnq import SDNQConfig
    from sdnq.common import use_torch_compile as triton_is_available
    from sdnq.loader import apply_sdnq_options_to_model
    
    model_id = "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32"
    
    print("\n" + "=" * 60)
    print("🚀 Loading Qwen-Image-Edit-2511 SDNQ UINT4 Pipeline")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Quantization: 4-bit UINT4 with SVD rank 32")
    print(f"Expected VRAM: ~12-15GB")
    print("-" * 60)
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    start_time = time.time()
    
    # Scheduler config for distilled model (4 steps)
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    
    # Load the SDNQ quantized pipeline
    print("Loading SDNQ quantized pipeline...")
    pipeline = diffusers.QwenImageEditPlusPipeline.from_pretrained(
        model_id,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    )
    
    # Enable INT8 MatMul for faster inference on NVIDIA GPUs
    if triton_is_available and torch.cuda.is_available():
        print("Enabling INT8 MatMul optimization...")
        pipeline.transformer = apply_sdnq_options_to_model(
            pipeline.transformer, use_quantized_matmul=True
        )
        pipeline.text_encoder = apply_sdnq_options_to_model(
            pipeline.text_encoder, use_quantized_matmul=True
        )
        print("✅ INT8 MatMul enabled!")
    
    # Enable CPU offload for memory efficiency
    print("Enabling model CPU offload...")
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
                  num_inference_steps: int = 4,
                  true_cfg_scale: float = 1.0,
                  seed: int = 42):
    """Run inference with the pipeline."""
    print("\n" + "=" * 60)
    print("🎨 Running Virtual Try-On Inference (SDNQ UINT4)")
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
        description="Qwen-Image-Edit-2511 SDNQ UINT4 Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This uses the SDNQ 4-bit quantized model which requires only ~12-15GB VRAM!

Install requirements:
    pip install sdnq

Examples:
    python test_qwen_sdnq.py --person person.jpg --cloth cloth.png
    python test_qwen_sdnq.py --person person.jpg --cloth cloth.png --steps 8
        """
    )
    parser.add_argument("--person", "-p", type=str, required=True,
                        help="Path to person image (with green mask)")
    parser.add_argument("--cloth", "-c", type=str, required=True,
                        help="Path to cloth/garment image")
    parser.add_argument("--output", "-o", type=str, default="vton_sdnq_output.png",
                        help="Output path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt (uses VTON prompt by default)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps (default: 4)")
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="True CFG scale (default: 1.0 for distilled model)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Default VTON prompt
    if args.prompt is None:
        args.prompt = """将图片 1 中的绿色遮罩区域仅用于判断服装属于上半身或下半身，不要将服装限制在遮罩范围内。

将图片 2 中的服装自然地穿戴到图片 1 中的人物身上，保持图片 2 中服装的完整形状、袖长和轮廓。无论图片 2 是单独的服装图还是人物穿着该服装的图，都应准确地转移服装，同时保留其原始面料质感、材质细节和颜色准确性。

确保图片 1 中人物的面部、头发和皮肤完全保持不变。光照与阴影应自然匹配图片 1 的环境，但服装的材质外观必须忠实于图片 2。

保持边缘平滑融合、阴影逼真，整体效果自然且不改变人物的身份特征。"""
    
    print("\n" + "👗" * 30)
    print("   QWEN VIRTUAL TRY-ON (SDNQ UINT4)")
    print("   4-bit Quantized - Low VRAM Mode!")
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
    
    # Load pipeline
    pipeline = load_pipeline_sdnq()
    
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
    print("   Using SDNQ 4-bit quantization")
    print("✅" * 30 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
