# LightX2V FP8 Baseline - Virtual Try-On

Fast VTON inference using LightX2V with Qwen-Image-Edit-2511.

## Performance
| Mode | Steps | Time (L40) | VRAM | Output |
|------|-------|------------|------|--------|
| FP8 | 4 | ~12-13s | ~35GB | 768x1024 |
| FP8+TeaCache | 4 | ~10s | ~35GB | 768x1024 |

## Quick Start

See `QUICKSTART.txt` for full instructions.

```bash
# Docker: lightx2v/lightx2v:25101501-cu124

git clone https://github.com/salahudeenofficial/try_og_pipeline.git
cd try_og_pipeline && git checkout fp8-baseline
chmod +x setup_lightx2v.sh && ./setup_lightx2v.sh
python test_lightx2v_vton.py --mode fp8 --person person.jpg --cloth cloth.png
```

## 🆕 GPU Server

A production-ready HTTP API for the VTON pipeline:

```bash
# Start the GPU server
cd gpu_server
pip install -r requirements.txt
./run.sh

# Server runs at http://localhost:8080
```

### Key Endpoints
- `POST /tryon` - Async inference (for backend integration)
- `POST /infer` - Sync inference (for frontend/testing)
- `GET /health` - Liveness probe
- `GET /test` - Readiness probe

See [gpu_server/README.md](gpu_server/README.md) for full documentation.

## 🆕 Frontend Tester

A modern web UI for testing the GPU server:

```bash
# Start the frontend
cd vton_frontend
python serve.py

# Open http://localhost:3000
```

Features:
- 🎨 Premium dark theme with glassmorphism
- 📤 Drag & drop image upload
- ⚙️ Configurable inference parameters
- 📊 Real-time server status

See [vton_frontend/README.md](vton_frontend/README.md) for details.

## Testing

### Test GPU Server (Mock Mode)
No GPU required - uses mock inference:

```bash
cd gpu_server
./run_mock.sh  # Start server in mock mode。

# In another terminal:
python tests/test_server.py --quick
```

### Test with Real Inference
Requires GPU and models:

```bash
cd gpu_server
./run.sh  # Start real server

# In another terminal:
python tests/test_server.py --inference
```

## Project Structure

```
try_og_pipeline/
├── test_lightx2v_vton.py     # Standalone inference script
├── setup_lightx2v.sh         # Setup script
├── person.jpg, cloth.png     # Test images
│
├── gpu_server/               # 🆕 HTTP API Server
│   ├── app.py                # FastAPI application
│   ├── configs/              # YAML configuration
│   ├── inference/            # Pipeline manager
│   ├── workflow/             # Editable workflow engine
│   └── tests/                # Test suite
│
└── vton_frontend/            # 🆕 Frontend Tester
    ├── index.html            # Main page
    ├── styles.css            # Premium styling
    ├── app.js                # Application logic
    └── serve.py              # Dev server
```

## Documentation
- `QUICKSTART.txt` - Quick start guide
- `GPU_SERVER_GUIDE.md` - API specification
- `gpu_server/README.md` - Server documentation
- `vton_frontend/README.md` - Frontend documentation
- `PROBLEMS_FACED.txt` - Known issues & fixes
