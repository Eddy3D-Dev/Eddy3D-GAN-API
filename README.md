---
title: Eddy3D GAN API
emoji: 🌬️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---
# Eddy3D GAN Wind Prediction API

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/SustainableUrbanSystemsLab/Eddy3D-GAN)

Gradio app that serves the GAN surrogate model for urban pedestrian-level wind flow prediction, accelerated with **ZeroGPU** (free H200 GPU).

**Live Endpoint:** `https://sustainableurbansystemslab-eddy3d-gan.hf.space`

## How it works

1. ONNX model is converted to PyTorch at startup via `onnx2torch`
2. Each inference call gets a ZeroGPU-allocated H200 via `@spaces.GPU`
3. Results are returned as JSON (wind speeds + PNG image, both base64-encoded)

## ONNX Model Hosting

The ONNX model file (`GAN-21-05-2023-23-Generative.onnx`, ~208 MB) is **not included** in this repository. It is downloaded at startup from:

```
https://huggingface.co/SustainableUrbanSystemsLab/UrbanWind-GAN/resolve/main/GAN-21-05-2023-23-Generative.onnx
```

## API Usage

### Python client

```python
from gradio_client import Client

client = Client("SustainableUrbanSystemsLab/Eddy3D-GAN")
result = client.predict(data_b64="...", api_name="/predict")
```

### cURL

```bash
# Step 1: Submit
EVENT_ID=$(curl -s -X POST \
  https://sustainableurbansystemslab-eddy3d-gan.hf.space/api/predict \
  -H "Content-Type: application/json" \
  -d '{"data": ["BASE64_DATA_HERE"]}' | jq -r '.event_id')

# Step 2: Fetch result
curl -N https://sustainableurbansystemslab-eddy3d-gan.hf.space/api/predict/$EVENT_ID
```

### Input format

Base64-encoded, gzip-compressed flat array of 786,432 float32 values (3 x 512 x 512), channel-first order (R, G, B), normalised to [-1, 1].

### Output format

```json
{
  "wind_speeds_b64": "...",
  "image_base64": "...",
  "width": 512,
  "height": 512
}
```

## Local Development

```bash
pip install -r requirements.txt

# Place your model file
cp /path/to/GAN-21-05-2023-23-Generative.onnx model.onnx

# Run (CPU mode locally — @spaces.GPU is a no-op outside HF Spaces)
python app.py
```
