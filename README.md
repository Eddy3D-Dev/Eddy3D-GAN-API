---
title: Eddy3D GAN API
emoji: 🌬️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---
# Eddy3D GAN Wind Prediction API

FastAPI service that serves the GAN surrogate model for urban pedestrian-level wind flow prediction.

**Live API Endpoint:** `https://sustainableurbansystemslab-eddy3d-gan.hf.space`

## ONNX Model Hosting

The ONNX model file (`GAN-21-05-2023-23-Generative.onnx`, ~208 MB) is **not included** in this repository. The container downloads it at startup from a URL you provide.

The model is hosted on Hugging Face:

```
https://huggingface.co/SustainableUrbanSystemsLab/UrbanWind-GAN/resolve/main/GAN-21-05-2023-23-Generative.onnx
```

`MODEL_URL` defaults to this URL if not provided, but you can override it via environment variable.

### Optional: SHA-256 Integrity Check

Generate a checksum and set `MODEL_SHA256` to verify downloads:

```bash
shasum -a 256 GAN-21-05-2023-23-Generative.onnx
```

## Local Development

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Place your model file
cp /path/to/GAN-21-05-2023-23-Generative.onnx model.onnx

# Run the server
uv run uvicorn api:app --reload --port 8000
```

## Deploying to Render

1. Push this repo to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect the GitHub repo
4. Render will auto-detect the `Dockerfile`
5. Set the following environment variables in the Render dashboard:

| Variable | Required | Example |
|----------|----------|---------|
| `MODEL_URL` | No | `https://huggingface.co/SustainableUrbanSystemsLab/UrbanWind-GAN/resolve/main/GAN-21-05-2023-23-Generative.onnx` |
| `MODEL_SHA256` | No | `abc123...` (hex digest for integrity check) |
| `RATE_LIMIT_REQUESTS` | No | `30` (default: 30 requests per window) |
| `RATE_LIMIT_WINDOW` | No | `minute` (options: second, minute, hour, day) |
| `ALLOWED_ORIGINS` | No | `*` (comma-separated CORS origins) |
| `LOG_LEVEL` | No | `INFO` |

Alternatively, use `render.yaml` for infrastructure-as-code deployment (Blueprint).

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check (not rate limited) |
| POST | `/predict` | Image upload → wind speeds JSON + output image (base64) |
| POST | `/predict_array` | Raw float array → wind speeds JSON + output image (base64) |
| POST | `/predict_image` | Image upload → output PNG stream (legacy) |

### `/predict_array` (primary endpoint)

Accepts a JSON body with a flat float array of 786,432 values (3 x 512 x 512), channel-first order (R, G, B), normalised to [-1, 1].

```json
{
  "data": [0.1, -0.5, ...]
}
```

Returns:

```json
{
  "wind_speeds": [0.5, 1.2, ...],
  "image_base64": "iVBORw0KGgo...",
  "width": 512,
  "height": 512
}
```

## Docker

```bash
docker build -t eddy3d-gan-api .
docker run -p 8000:8000 -e MODEL_URL="https://huggingface.co/SustainableUrbanSystemsLab/UrbanWind-GAN/resolve/main/GAN-21-05-2023-23-Generative.onnx" eddy3d-gan-api
```
