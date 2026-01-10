# GPU Server API Guide

A concise guide for building a GPU inference server. This document focuses on the API endpoints, their behavior, and the overall request flow.

---

## Overview

The GPU Server is a stateless HTTP API that processes virtual try-on inference requests asynchronously. Each server instance handles one job at a time and delivers results via HTTP callbacks.

### Request Flow

```
1. CPU Bridge → POST /tryon → GPU Server
2. GPU Server → Returns 202 Accepted immediately
3. GPU Server → Processes inference in background
4. GPU Server → POST callback to Asset Service with result image
5. GPU Server → POST /job_complete to Load Balancer
```

---

## API Endpoints

### POST /tryon

**Purpose**: Main inference endpoint. Accepts a job and processes it asynchronously.

**Authentication**: Requires `X-Internal-Auth` header

**Request Format**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string (UUID) | ✅ | Unique job identifier |
| `user_id` | string | ✅ | User identifier |
| `session_id` | string (UUID) | ✅ | Session identifier |
| `provider` | string | ✅ | Model provider (e.g., `"qwen"`, `"lightxv"`) |
| `masked_user_image` | file (image/png) | ✅ | Masked user image |
| `garment_image` | file (image/png) | ✅ | Garment image |
| `config` | string (JSON) | ✅ | Model config: `{"seed": 42, "steps": 4, "cfg": 1.0}` |

**Responses**:

| Status | Condition | Response Body |
|--------|-----------|---------------|
| `202 Accepted` | Job accepted | `{"status": "accepted", "job_id": "...", "message": "Job queued for processing"}` |
| `429 Too Many Requests` | GPU busy | `{"status": "busy", "message": "GPU server is currently busy", "retry_after": 5}` |
| `401 Unauthorized` | Invalid auth token | Error message |
| `400 Bad Request` | Invalid provider or missing fields | Error message |

**Behavior**:
1. Validate the `X-Internal-Auth` header
2. Check if GPU is available (not busy with another job)
3. If busy → return `429` with `Retry-After: 5` header
4. If available → mark GPU as busy, return `202` immediately
5. Process inference in background (never blocks the HTTP response)
6. On completion → send result to Asset Service callback URL
7. On completion → notify Load Balancer via `/job_complete`
8. Mark GPU as free

---

### GET /health

**Purpose**: Liveness probe for load balancer. Indicates if the server is running.

**Authentication**: None required

**Response** (always `200 OK`):
```json
{
  "status": "ok",
  "model_loaded": true,
  "node_id": "gpu-node-1"
}
```

**Behavior**:
- Must respond within 300ms
- Returns `model_loaded: true` only after models are fully loaded
- Used by Load Balancer to detect if node is alive

---

### GET /test

**Purpose**: Readiness probe for load balancer. Indicates if the server is ready to accept jobs.

**Authentication**: None required

**Response when ready** (`200 OK`):
```json
{
  "status": "hot",
  "model_loaded": true,
  "node_id": "gpu-node-1",
  "model_type": "qwen"
}
```

**Response when not ready** (`200 OK`):
```json
{
  "status": "loading",
  "model_loaded": false,
  "node_id": "gpu-node-1"
}
```

**Behavior**:
- Must respond within 300ms
- `status: "hot"` means ready to accept inference jobs
- `status: "loading"` means models are still loading
- Load Balancer only routes jobs to nodes with `status: "hot"`

---

### GET /gpu/status

**Purpose**: Get GPU busy state for load balancer routing decisions.

**Authentication**: Requires `X-Internal-Auth` header

**Response** (`200 OK`):
```json
{
  "node_id": "gpu-node-1",
  "busy": false,
  "current_job_id": null,
  "queue_length": 0
}
```

| Field | Type | Description |
|-------|------|-------------|
| `node_id` | string | Unique identifier for this GPU node |
| `busy` | boolean | `true` if currently processing a job |
| `current_job_id` | string or null | ID of job being processed |
| `queue_length` | integer | Number of jobs processed (or queued) |

**Behavior**:
- Returns current GPU state
- Load Balancer uses this to decide which node to route jobs to

---

### GET /version

**Purpose**: Get version and model information.

**Authentication**: None required

**Response** (`200 OK`):
```json
{
  "model_type": "qwen",
  "model_version": "1.0.0",
  "backend": "comfyui-python",
  "git_commit": "abc1234",
  "node_id": "gpu-node-1"
}
```

---

### GET /metrics

**Purpose**: Prometheus-style metrics for monitoring.

**Authentication**: None required

**Response** (`200 OK`):
```json
{
  "vton_inference_count": 150,
  "vton_inference_latency_ms": 2500.5,
  "vton_inference_errors_total": 2,
  "gpu_memory_used_bytes": 8589934592,
  "gpu_utilization_percent": 85.5
}
```

---

## Callback Specifications

### Asset Service Callback (Result Delivery)

After inference completes (success or failure), the GPU server sends results to the Asset Service.

**URL**: Configured in `config.yaml` → `asset_service.callback_url`

**Method**: `POST`

**Format**: `multipart/form-data`

**Headers**:
```
X-Internal-Auth: <GPU_TO_ASSET_SECRET>
Content-Type: multipart/form-data
```

**Form Data**:

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Original job identifier |
| `user_id` | string | User identifier |
| `session_id` | string | Session identifier |
| `provider` | string | Model provider (e.g., `"qwen"`) |
| `node_id` | string | GPU node identifier |
| `model_version` | string | Model version used |
| `inference_time_ms` | string | Inference duration in milliseconds |
| `output_image` | file | Result image (PNG) - only on success |
| `error` | string | Error message - only on failure |

**Expected Response**: `200 OK`

**Retry Behavior**:
- Retries up to 3 times (configurable)
- Exponential backoff: 1s, 2s, 4s
- If all retries fail, error is logged but job is still marked complete

---

### Load Balancer Callback (Job Complete Notification)

After every job (success or failure), the GPU server notifies the Load Balancer.

**URL**: Configured in `config.yaml` → `load_balancer.url` + `/job_complete`

**Method**: `POST`

**Format**: `application/json`

**Headers**:
```
Content-Type: application/json
X-Internal-Auth: <LB_AUTH_TOKEN>  (optional)
```

**Request Body**:
```json
{
  "node_id": "gpu-node-1",
  "job_id": "dd1283e6-91a9-4f40-851e-8687a5d557dd",
  "metadata": {}
}
```

**Behavior**:
- Non-blocking (5 second timeout)
- If Load Balancer URL not configured, callback is skipped
- Failures are logged but don't affect job completion

---

## Server Behavior Rules

### Startup Behavior

1. Load configuration from `config.yaml`
2. Load all models into GPU memory
3. Reject all traffic (except `/health` and `/test`) with `503 Service Unavailable` until models are loaded
4. Once models are loaded, begin accepting jobs

### Request Handling

1. **Single Job at a Time**: Only one inference job can run at a time per server instance
2. **Immediate Response**: Always return `202` or `429` immediately, never block waiting for inference
3. **Background Processing**: Inference runs in a background task
4. **Callback Delivery**: Results are NEVER returned in the HTTP response, always sent via callback

### Error Handling

| Scenario | Behavior |
|----------|----------|
| GPU busy | Return `429 Too Many Requests` with `Retry-After: 5` header |
| Models not loaded | Return `503 Service Unavailable` |
| Invalid auth | Return `401 Unauthorized` |
| Inference fails | Send error callback to Asset Service, mark job complete |
| Callback fails | Retry 3 times, then log error and continue |

---

## Configuration

All runtime configuration is in `configs/config.yaml`:

```yaml
server:
  node_id: "gpu-node-1"           # Unique identifier for this node

security:
  internal_auth_token: "SECRET"   # Token for incoming requests from CPU Bridge

asset_service:
  callback_url: "http://asset-service:9009/v1/vton/result"
  internal_auth_token: "SECRET"   # Token for outgoing callbacks
  timeout: 10                     # Callback timeout (seconds)
  retries: 3                      # Number of callback retries

model:
  model_type: "qwen"              # Model type identifier
  model_version: "1.0.0"          # Model version string
  device: "cuda"                  # Device to use

load_balancer:
  url: "http://load-balancer:9005"    # Optional
  internal_auth_token: "SECRET"       # Optional

logging:
  level: "INFO"
```

---

## Summary Table

| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/tryon` | POST | ✅ | Submit inference job |
| `/health` | GET | ❌ | Liveness probe |
| `/test` | GET | ❌ | Readiness probe |
| `/gpu/status` | GET | ✅ | GPU busy state |
| `/version` | GET | ❌ | Version info |
| `/metrics` | GET | ❌ | Prometheus metrics |
