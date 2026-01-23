# Config Field Explanation

## Overview

The `config` field in the `/tryon` request is a **JSON string** that controls inference parameters for the virtual try-on generation. It allows you to customize how the model generates the output image.

## Format

```json
{
  "seed": 42,
  "steps": 4,
  "cfg": 1.0
}
```

**Important:** This must be sent as a **string** in the form data, not as a JSON object.

## Field Details

### 1. `seed` (integer, optional)

**Purpose:** Random seed for reproducible results

**Default:** `42` (from `config.yaml`)

**Range:** Any integer (typically 0 to 2,147,483,647)

**How it works:**
- Controls the random number generator used during inference
- Same seed + same inputs = same output (deterministic)
- Different seed = different output (variation)

**Example:**
```json
{"seed": 42}   // Always produces the same result
{"seed": 123}  // Different result
{"seed": 999}  // Another different result
```

**Use cases:**
- **Reproducibility:** Use the same seed to get identical results
- **Variation:** Change seed to generate different variations
- **Debugging:** Use fixed seed to test consistency

---

### 2. `steps` (integer, optional)

**Purpose:** Number of inference steps (denoising iterations)

**Default:** `4` (from `config.yaml`)

**Typical Range:** 
- **4 steps** - Fast, optimized for Lightning models (recommended)
- **40 steps** - Slower, higher quality (for base models)

**How it works:**
- More steps = more denoising iterations = potentially better quality
- Fewer steps = faster inference = lower quality
- The model uses a 4-step distillation, so 4 steps is optimal

**Example:**
```json
{"steps": 4}   // Fast, good quality (recommended for FP8/LoRA)
{"steps": 40}  // Slower, best quality (for base model)
```

**Important Notes:**
- **4 steps** is recommended for FP8 and LoRA modes (fast + good quality)
- **40 steps** is for base model mode (slower but highest quality)
- The server is configured with `default_steps: 4` for optimal performance

**Performance Impact:**
- 4 steps: ~12-13 seconds per inference
- 40 steps: ~60+ seconds per inference

---

### 3. `cfg` (float, optional)

**Purpose:** Classifier-Free Guidance (CFG) scale

**Default:** `1.0` (from `config.yaml`)

**Typical Range:** `0.1` to `20.0`

**How it works:**
- Controls how strongly the model follows the input prompt/guidance
- Lower values = more creative/flexible output
- Higher values = more strict adherence to guidance
- For VTON (virtual try-on), `1.0` is typically optimal

**Example:**
```json
{"cfg": 1.0}   // Balanced (recommended)
{"cfg": 0.5}   // More flexible/creative
{"cfg": 2.0}   // More strict/adherent
```

**Use cases:**
- **1.0** - Standard, balanced guidance (recommended)
- **< 1.0** - More variation, less strict
- **> 1.0** - More strict adherence to input

**Note:** For virtual try-on, the model doesn't use text prompts in the same way as text-to-image models, so CFG has less impact. `1.0` is the recommended value.

---

## Complete Examples

### Minimal Config (uses all defaults)
```json
{}
```
This will use:
- `seed: 42`
- `steps: 4`
- `cfg: 1.0`

### Standard Config (recommended)
```json
{
  "seed": 42,
  "steps": 4,
  "cfg": 1.0
}
```

### Custom Config (for variation)
```json
{
  "seed": 12345,
  "steps": 4,
  "cfg": 1.0
}
```

### High Quality Config (slower)
```json
{
  "seed": 42,
  "steps": 40,
  "cfg": 1.0
}
```

---

## How to Send in Request

### As JSON String (correct)
```python
import json

config = json.dumps({
    "seed": 42,
    "steps": 4,
    "cfg": 1.0
})

# Send as string in form data
data = {
    "config": config,  # String, not dict!
    ...
}
```

### cURL Example
```bash
-F "config={\"seed\": 42, \"steps\": 4, \"cfg\": 1.0}"
```

### Python requests Example
```python
import json

data = {
    "job_id": "job-123",
    "user_id": "user-abc",
    "session_id": "session-xyz",
    "provider": "qwen",
    "config": json.dumps({"seed": 42, "steps": 4, "cfg": 1.0})  # JSON string!
}

files = {
    "masked_user_image": open("person.png", "rb"),
    "garment_image": open("garment.png", "rb")
}

response = requests.post(url, headers=headers, data=data, files=files)
```

---

## Default Values

If you don't specify a field, the server uses defaults from `config.yaml`:

| Field | Default Value | Location |
|-------|--------------|----------|
| `seed` | `42` | `config.model.default_seed` |
| `steps` | `4` | `config.model.default_steps` |
| `cfg` | `1.0` | `config.model.default_cfg` |

---

## Recommendations

### For Production Use
```json
{
  "seed": 42,
  "steps": 4,
  "cfg": 1.0
}
```
- Fast inference (~12-13 seconds)
- Good quality results
- Consistent output

### For Testing/Development
```json
{
  "seed": 123,
  "steps": 4,
  "cfg": 1.0
}
```
- Use different seeds to see variations
- Keep steps at 4 for speed

### For Maximum Quality (if needed)
```json
{
  "seed": 42,
  "steps": 40,
  "cfg": 1.0
}
```
- Slower (~60+ seconds)
- Best quality
- Only if using base model mode

---

## Error Handling

### Invalid JSON
If the config string is not valid JSON, you'll get a `400 Bad Request` error.

### Missing Fields
All fields are optional. Missing fields will use defaults from `config.yaml`.

### Invalid Values
- `steps` must be a positive integer
- `cfg` should be a positive float
- `seed` can be any integer

---

## Summary

| Parameter | Purpose | Default | Recommended | Impact |
|-----------|---------|---------|-------------|--------|
| `seed` | Reproducibility | 42 | 42 | Controls randomness |
| `steps` | Quality vs Speed | 4 | 4 | More steps = slower + better |
| `cfg` | Guidance strength | 1.0 | 1.0 | Less impact for VTON |

**Best Practice:** Use `{"seed": 42, "steps": 4, "cfg": 1.0}` for optimal balance of speed and quality.
