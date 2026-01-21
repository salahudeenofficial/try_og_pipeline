# GPU Server Quick Start Guide

## 🚀 Quick Setup (All-in-One)

```bash
cd gpu_server
./setup_and_start.sh
```

This will:
1. Install all requirements (parent + GPU server)
2. Verify configuration
3. Start server on port 8000

---

## 📝 Manual Setup

### 1. Configure Callbacks

Edit `configs/config.yaml`:

```yaml
asset_service:
  callback_url: "http://YOUR_BACKEND_HOST:PORT/v1/vton/result"  # Your backend endpoint
  internal_auth_token: "your-secret-token"                      # Token for callbacks
  timeout: 10                                                    # Callback timeout (seconds)
  retries: 3                                                     # Number of retries
  retry_backoff: [1, 2, 4]                                      # Retry delays
```

**Current Configuration:**
- Callback URL: `http://65.0.6.48:9009/v1/vton/result`
- Auth Token: `supersecret-internal-token`

### 2. Install Requirements

```bash
# From project root
cd /home/fashionx/try_og_pipeline

# Install parent requirements (PyTorch, transformers, etc.)
pip install -r requirements.txt

# Install GPU server requirements
cd gpu_server
pip install -r requirements.txt
```

### 3. Start Server on Port 8000

**Option A: Using the startup script**
```bash
cd gpu_server
./run.sh
```

**Option B: Direct uvicorn command**
```bash
cd gpu_server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```

**Option C: Using the all-in-one script**
```bash
cd gpu_server
./setup_and_start.sh
```

---

## ✅ Verify Server is Running

```bash
# Health check
curl http://localhost:8000/health

# Expected response: {"status": "ok"}
```

---

## 📡 Callback Details

When inference completes, the server sends a POST request to your callback URL:

**Endpoint:** `POST {callback_url}`

**Headers:**
```
X-Internal-Auth: {internal_auth_token}
Content-Type: multipart/form-data
```

**Form Data:**
- `job_id` - Job identifier
- `user_id` - User identifier  
- `session_id` - Session identifier
- `provider` - Model provider (e.g., "qwen")
- `node_id` - GPU node identifier
- `model_version` - Model version used
- `inference_time_ms` - Inference duration
- `output_image` - Result PNG image (on success)
- `error` - Error message (on failure)

**Expected Response:** `200 OK`

---

## 🔧 Configuration Files

- **Main Config:** `gpu_server/configs/config.yaml`
- **Dev Config:** `gpu_server/configs/config.dev.yaml` (for testing)

---

## 📋 Current Settings

- **Port:** 8000 ✅
- **Host:** 0.0.0.0 (all interfaces)
- **Callback URL:** http://65.0.6.48:9009/v1/vton/result
- **Default Steps:** 4 ✅
- **Default Mode:** fp8

---

## 🐛 Troubleshooting

**Port already in use:**
```bash
# Check what's using port 8000
lsof -i :8000
# Kill the process or change port in config.yaml
```

**Missing dependencies:**
```bash
# Reinstall requirements
pip install -r requirements.txt
cd gpu_server && pip install -r requirements.txt
```

**Callback not working:**
- Verify callback URL is reachable from GPU server
- Check auth token matches backend expectations
- Check server logs for callback errors
