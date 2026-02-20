"""Download ONNX model from cloud storage if not already present."""

import hashlib
import os
import sys

import requests

MODEL_URL = os.environ.get("MODEL_URL", "")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
MODEL_SHA256 = os.environ.get("MODEL_SHA256", "")


def download() -> None:
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return

    if not MODEL_URL:
        print(f"ERROR: MODEL_URL not set and model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading model from {MODEL_URL} ...")
    response = requests.get(MODEL_URL, stream=True, timeout=600)
    response.raise_for_status()

    os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)

    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
    print()

    # Optional SHA-256 integrity check
    if MODEL_SHA256:
        h = hashlib.sha256()
        with open(MODEL_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        if h.hexdigest() != MODEL_SHA256:
            os.remove(MODEL_PATH)
            print(f"ERROR: SHA-256 mismatch! Expected {MODEL_SHA256}, got {h.hexdigest()}", file=sys.stderr)
            sys.exit(1)
        print("SHA-256 verified.")

    print(f"Model downloaded to {MODEL_PATH}")


if __name__ == "__main__":
    download()
