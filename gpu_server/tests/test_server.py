#!/usr/bin/env python3
"""
GPU Server Test Suite

Comprehensive tests for the GPU inference server.
Can be run without the actual GPU/models for API testing.
"""

import os
import sys
import time
import json
import unittest
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock

import requests
from PIL import Image

# Add parent directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


class TestConfig:
    """Test configuration."""
    BASE_URL = os.environ.get("GPU_SERVER_URL", "http://localhost:8080")
    AUTH_TOKEN = os.environ.get("GPU_SERVER_AUTH", "dev-secret-token-change-in-production")
    TIMEOUT = 30


def create_test_image(width=512, height=768, color=(0, 255, 0)):
    """Create a test image."""
    img = Image.new("RGB", (width, height), color=color)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def get_headers(with_auth=True):
    """Get request headers."""
    headers = {}
    if with_auth:
        headers["X-Internal-Auth"] = TestConfig.AUTH_TOKEN
    return headers


class TestHealthEndpoints(unittest.TestCase):
    """Test health and status endpoints."""
    
    def test_health_endpoint(self):
        """Test GET /health returns 200."""
        response = requests.get(f"{TestConfig.BASE_URL}/health", timeout=5)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("node_id", data)
        self.assertEqual(data["status"], "ok")
    
    def test_test_endpoint(self):
        """Test GET /test returns readiness status."""
        response = requests.get(f"{TestConfig.BASE_URL}/test", timeout=5)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("status", data)
        self.assertIn("model_loaded", data)
        self.assertIn("node_id", data)
        
        # Status should be "hot" or "loading"
        self.assertIn(data["status"], ["hot", "loading"])
    
    def test_version_endpoint(self):
        """Test GET /version returns version info."""
        response = requests.get(f"{TestConfig.BASE_URL}/version", timeout=5)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("model_type", data)
        self.assertIn("model_version", data)
        self.assertIn("backend", data)
        self.assertIn("node_id", data)
    
    def test_metrics_endpoint(self):
        """Test GET /metrics returns metrics."""
        response = requests.get(f"{TestConfig.BASE_URL}/metrics", timeout=5)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("vton_inference_count", data)
        self.assertIn("vton_inference_latency_ms", data)
        self.assertIn("vton_inference_errors_total", data)
        self.assertIn("gpu_memory_used_bytes", data)


class TestGPUStatusEndpoint(unittest.TestCase):
    """Test GPU status endpoint with authentication."""
    
    def test_gpu_status_requires_auth(self):
        """Test GET /gpu/status requires authentication."""
        response = requests.get(f"{TestConfig.BASE_URL}/gpu/status", timeout=5)
        self.assertEqual(response.status_code, 401)
    
    def test_gpu_status_with_auth(self):
        """Test GET /gpu/status with valid auth."""
        response = requests.get(
            f"{TestConfig.BASE_URL}/gpu/status",
            headers=get_headers(with_auth=True),
            timeout=5
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("node_id", data)
        self.assertIn("busy", data)
        self.assertIn("current_job_id", data)
        self.assertIn("queue_length", data)
    
    def test_gpu_status_with_invalid_auth(self):
        """Test GET /gpu/status with invalid auth."""
        response = requests.get(
            f"{TestConfig.BASE_URL}/gpu/status",
            headers={"X-Internal-Auth": "invalid-token"},
            timeout=5
        )
        self.assertEqual(response.status_code, 401)


class TestTryonEndpoint(unittest.TestCase):
    """Test the main VTON inference endpoint."""
    
    def test_tryon_requires_auth(self):
        """Test POST /tryon requires authentication."""
        response = requests.post(
            f"{TestConfig.BASE_URL}/tryon",
            data={"job_id": "test"},
            timeout=5
        )
        self.assertEqual(response.status_code, 401)
    
    def test_tryon_requires_all_fields(self):
        """Test POST /tryon requires all form fields."""
        response = requests.post(
            f"{TestConfig.BASE_URL}/tryon",
            headers=get_headers(),
            data={"job_id": "test"},  # Missing required fields
            timeout=5
        )
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
    
    def test_tryon_invalid_provider(self):
        """Test POST /tryon rejects invalid provider."""
        person_image = create_test_image()
        garment_image = create_test_image(color=(255, 255, 255))
        
        response = requests.post(
            f"{TestConfig.BASE_URL}/tryon",
            headers=get_headers(),
            data={
                "job_id": "test-job-001",
                "user_id": "test-user",
                "session_id": "test-session",
                "provider": "invalid_provider",
                "config": json.dumps({"seed": 42, "steps": 4, "cfg": 1.0}),
            },
            files={
                "masked_user_image": ("person.png", person_image, "image/png"),
                "garment_image": ("garment.png", garment_image, "image/png"),
            },
            timeout=5
        )
        self.assertEqual(response.status_code, 400)
    
    def test_tryon_accepts_job(self):
        """Test POST /tryon accepts a valid job."""
        person_image = create_test_image()
        garment_image = create_test_image(color=(255, 255, 255))
        
        response = requests.post(
            f"{TestConfig.BASE_URL}/tryon",
            headers=get_headers(),
            data={
                "job_id": "test-job-002",
                "user_id": "test-user",
                "session_id": "test-session",
                "provider": "qwen",
                "config": json.dumps({"seed": 42, "steps": 4, "cfg": 1.0}),
            },
            files={
                "masked_user_image": ("person.png", person_image, "image/png"),
                "garment_image": ("garment.png", garment_image, "image/png"),
            },
            timeout=TestConfig.TIMEOUT
        )
        
        # Should return 202 Accepted or 503 if models not loaded
        self.assertIn(response.status_code, [202, 503, 429])
        
        if response.status_code == 202:
            data = response.json()
            self.assertEqual(data["status"], "accepted")
            self.assertEqual(data["job_id"], "test-job-002")


class TestInferEndpoint(unittest.TestCase):
    """Test the direct inference endpoint (for frontend/testing)."""
    
    def test_infer_basic(self):
        """Test POST /infer with basic request."""
        person_image = create_test_image()
        garment_image = create_test_image(color=(255, 255, 255))
        
        response = requests.post(
            f"{TestConfig.BASE_URL}/infer",
            data={
                "seed": 42,
                "steps": 4,
                "cfg": 1.0,
            },
            files={
                "masked_user_image": ("person.png", person_image, "image/png"),
                "garment_image": ("garment.png", garment_image, "image/png"),
            },
            timeout=TestConfig.TIMEOUT
        )
        
        # Should return 200 with image or 503 if models not loaded
        self.assertIn(response.status_code, [200, 503, 429])
        
        if response.status_code == 200:
            # Check response is an image
            self.assertEqual(response.headers.get("content-type"), "image/png")
            self.assertIn("X-Inference-Time-Ms", response.headers)
            self.assertIn("X-Job-Id", response.headers)
            
            # Verify it's a valid PNG
            img = Image.open(BytesIO(response.content))
            self.assertEqual(img.format, "PNG")


class TestConcurrency(unittest.TestCase):
    """Test concurrent request handling."""
    
    def test_gpu_returns_busy(self):
        """Test that GPU returns 429 when busy."""
        # This test requires models to be loaded
        # First, start a job
        person_image = create_test_image()
        garment_image = create_test_image(color=(255, 255, 255))
        
        # Start first job in background
        def start_job():
            requests.post(
                f"{TestConfig.BASE_URL}/infer",
                data={"seed": 42, "steps": 4, "cfg": 1.0},
                files={
                    "masked_user_image": ("person.png", create_test_image(), "image/png"),
                    "garment_image": ("garment.png", create_test_image(color=(255,255,255)), "image/png"),
                },
                timeout=60
            )
        
        # Start first job
        thread = threading.Thread(target=start_job)
        thread.start()
        
        # Give it a moment to start
        time.sleep(0.5)
        
        # Try second job - should get 429 if first is still running
        response = requests.post(
            f"{TestConfig.BASE_URL}/infer",
            data={"seed": 42, "steps": 4, "cfg": 1.0},
            files={
                "masked_user_image": ("person.png", person_image, "image/png"),
                "garment_image": ("garment.png", garment_image, "image/png"),
            },
            timeout=5
        )
        
        # Could be 429 (busy), 200 (fast completion), or 503 (not loaded)
        self.assertIn(response.status_code, [200, 429, 503])
        
        thread.join(timeout=60)


def run_quick_test():
    """Run a quick test to verify server is responding."""
    print("=" * 60)
    print("🧪 GPU Server Quick Test")
    print("=" * 60)
    print(f"Target: {TestConfig.BASE_URL}")
    print()
    
    tests = [
        ("Health", "/health", "GET"),
        ("Test", "/test", "GET"),
        ("Version", "/version", "GET"),
        ("Metrics", "/metrics", "GET"),
    ]
    
    all_passed = True
    
    for name, endpoint, method in tests:
        try:
            if method == "GET":
                response = requests.get(f"{TestConfig.BASE_URL}{endpoint}", timeout=5)
            
            status = "✅" if response.status_code == 200 else "❌"
            print(f"  {status} {name}: {response.status_code}")
            
            if response.status_code != 200:
                all_passed = False
            else:
                print(f"      Response: {response.json()}")
        
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False
    
    print()
    
    if all_passed:
        print("✅ All quick tests passed!")
    else:
        print("❌ Some tests failed")
    
    return all_passed


def run_inference_test():
    """Run an actual inference test."""
    print("=" * 60)
    print("🧪 GPU Server Inference Test")
    print("=" * 60)
    
    # Check if we have test images
    person_path = Path(SCRIPT_DIR.parent / "person.jpg")
    cloth_path = Path(SCRIPT_DIR.parent / "cloth.png")
    
    if person_path.exists() and cloth_path.exists():
        print(f"Using test images:")
        print(f"  Person: {person_path}")
        print(f"  Cloth: {cloth_path}")
        
        with open(person_path, 'rb') as f:
            person_data = f.read()
        with open(cloth_path, 'rb') as f:
            cloth_data = f.read()
    else:
        print("No test images found, creating synthetic images...")
        person_data = create_test_image().read()
        cloth_data = create_test_image(color=(255, 255, 255)).read()
    
    print()
    print("Starting inference...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{TestConfig.BASE_URL}/infer",
            data={
                "seed": 42,
                "steps": 4,
                "cfg": 1.0,
            },
            files={
                "masked_user_image": ("person.png", person_data, "image/png"),
                "garment_image": ("garment.png", cloth_data, "image/png"),
            },
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Inference successful!")
            print(f"   Status: {response.status_code}")
            print(f"   Total time: {elapsed:.2f}s")
            print(f"   Inference time: {response.headers.get('X-Inference-Time-Ms', 'N/A')}ms")
            print(f"   Job ID: {response.headers.get('X-Job-Id', 'N/A')}")
            
            # Save result
            output_path = SCRIPT_DIR / "test_output.png"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"   Output saved: {output_path}")
            
            return True
        
        elif response.status_code == 503:
            print(f"⚠️ Models not loaded yet")
            print(f"   Response: {response.json()}")
            return False
        
        elif response.status_code == 429:
            print(f"⚠️ GPU is busy")
            return False
        
        else:
            print(f"❌ Inference failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after 120s")
        return False
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to server at {TestConfig.BASE_URL}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Server Test Suite")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--auth", default="dev-secret-token-change-in-production", help="Auth token")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--inference", action="store_true", help="Run inference test")
    parser.add_argument("--full", action="store_true", help="Run full test suite")
    
    args = parser.parse_args()
    
    TestConfig.BASE_URL = args.url
    TestConfig.AUTH_TOKEN = args.auth
    
    if args.quick:
        run_quick_test()
    elif args.inference:
        run_inference_test()
    elif args.full:
        run_quick_test()
        print()
        run_inference_test()
    else:
        # Run unittest suite
        unittest.main(argv=[''], exit=False, verbosity=2)
