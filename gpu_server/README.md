# GPU Server

A stateless HTTP API for GPU-based virtual try-on inference using LightX2V.

## Overview

The GPU Server processes virtual try-on inference requests asynchronously. Each server instance handles one job at a time and delivers results via HTTP callbacks (for backend integration) or synchronously (for frontend/testing).

## Architecture

```
vton_frontend (testing) ──► /infer ──────► GPU Server ──► Result Image
                                              ▲
CPU Bridge (production) ──► /tryon ───────────┘
                                              │
                                              ├──► Asset Service Callback
                                              └──► Load Balancer Notification
```

## Quick Start

### 1. Install Dependencies

```bash
cd gpu_server
pip install -r requirements.txt
```

### 2. Configure

Edit `configs/config.yaml` to configure:
- Server settings (port, node ID)
- Security (auth tokens)
- Model settings (mode, resolution)
- Callback URLs (for production)

### 3. Run the Server

```bash
# Using the startup script
chmod +x run.sh
./run.sh

# Or directly with uvicorn
python -m uvicorn app:app --host 0.0.0.0 --port 8080
```

### 4. Test

```bash
# Quick API test
python tests/test_server.py --quick

# Full inference test
python tests/test_server.py --inference

# Run all unit tests
python tests/test_server.py
```

## API Endpoints

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/tryon` | POST | ✅ | Submit async inference job (for backend) |
| `/infer` | POST | ⚡ | Direct sync inference (for frontend/testing) |
| `/health` | GET | ❌ | Liveness probe |
| `/test` | GET | ❌ | Readiness probe |
| `/gpu/status` | GET | ✅ | GPU busy state |
| `/version` | GET | ❌ | Version info |
| `/metrics` | GET | ❌ | Prometheus metrics |

## Configuration

All configuration is in `configs/config.yaml`:

```yaml
server:
  node_id: "gpu-node-1"
  port: 8080

security:
  internal_auth_token: "your-secret-token"
  require_auth: true

model:
  default_mode: "fp8"        # fp8, lora, or base
  default_steps: 4
  default_resolution: "720p"

# Easily editable workflow settings
workflow:
  preprocess:
    resize_to_resolution: true
    maintain_aspect_ratio: true
  postprocess:
    create_comparison: false
  advanced:
    warmup_on_startup: true
    clear_cache_between_jobs: true
```

## Workflow Customization

The workflow is designed to be easily editable. You can:

1. **Modify config settings** - Change resolution, enable/disable features
2. **Add workflow steps** - Create new `WorkflowStep` classes
3. **Reorder steps** - Use `workflow.insert_step()` and `workflow.remove_step()`

Example custom workflow step:

```python
from workflow.engine import WorkflowStep, JobContext

class MyCustomStep(WorkflowStep):
    @property
    def name(self) -> str:
        return "MyCustomStep"
    
    def execute(self, context: JobContext) -> JobContext:
        # Your custom logic here
        return context
```

## Project Structure

```
gpu_server/
├── app.py                    # FastAPI application
├── run.sh                    # Startup script
├── requirements.txt          # Dependencies
├── configs/
│   └── config.yaml           # Configuration file
├── core/
│   ├── __init__.py
│   └── config.py             # Configuration loader
├── inference/
│   ├── __init__.py
│   └── pipeline_manager.py   # LightX2V pipeline
├── workflow/
│   ├── __init__.py
│   └── engine.py             # Workflow engine & steps
└── tests/
    ├── __init__.py
    └── test_server.py        # Test suite
```

## Production Deployment

For production, configure:

1. **Asset Service Callback** - Where to send inference results
2. **Load Balancer** - For job routing notifications
3. **Security** - Enable auth and use strong tokens
4. **Logging** - Set appropriate log level

```yaml
security:
  internal_auth_token: "${GPU_SERVER_AUTH_TOKEN}"
  require_auth: true

asset_service:
  callback_url: "http://asset-service:9009/v1/vton/result"
  internal_auth_token: "${ASSET_SERVICE_TOKEN}"

load_balancer:
  url: "http://load-balancer:9005"
```

## Monitoring

The `/metrics` endpoint provides Prometheus-compatible metrics:

- `vton_inference_count` - Total inference count
- `vton_inference_latency_ms` - Average latency
- `vton_inference_errors_total` - Error count
- `gpu_memory_used_bytes` - GPU memory usage
- `gpu_utilization_percent` - GPU utilization

## Troubleshooting

### Models not loading
- Check model paths in `pipeline_manager.py`
- Ensure models are downloaded: `huggingface-cli download Qwen/Qwen-Image-Edit-2511`

### GPU busy (429 errors)
- The server processes one job at a time
- Implement retry logic with exponential backoff

### Connection refused
- Check if the server is running
- Verify firewall settings
- Check the port configuration
