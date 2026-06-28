"""
app.py — Flask backend for PneumoScanAI
Serves the fine-tuned Vision Transformer (ViT) model:
  pneumonia_vit_checkpoint.pt

Place that checkpoint file in this same folder before starting the server.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import io
import base64
import os
import traceback
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import ViTForImageClassification

app = Flask(__name__)
CORS(app)

# ── Configuration ─────────────────────────────────────────────────────────────
VIT_CHECKPOINT = os.environ.get("VIT_CHECKPOINT", "pneumonia_vit_checkpoint.pt")

model         = None
vit_threshold = 0.5     # overwritten with the val-tuned value from the checkpoint
IMG_SIZE      = 224
IMG_MEAN      = [0.5, 0.5, 0.5]
IMG_STD       = [0.5, 0.5, 0.5]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loader ────────────────────────────────────────────────────────────────────
def load_vit():
    global model, vit_threshold, IMG_SIZE, IMG_MEAN, IMG_STD

    print(f"[INFO] Device: {DEVICE}")
    checkpoint = torch.load(VIT_CHECKPOINT, map_location=DEVICE)

    model_name    = checkpoint.get("model_name", "google/vit-base-patch16-224-in21k")
    vit_threshold = checkpoint.get("threshold", 0.5)
    IMG_SIZE      = checkpoint.get("img_size",  224)
    IMG_MEAN      = checkpoint.get("img_mean",  [0.5, 0.5, 0.5])
    IMG_STD       = checkpoint.get("img_std",   [0.5, 0.5, 0.5])

    vit = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    # Rebuild the exact same head used during training
    vit.classifier = nn.Sequential(
        nn.Linear(vit.config.hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1)
    )
    vit.load_state_dict(checkpoint["model_state_dict"])
    vit.to(DEVICE)
    vit.eval()

    model = vit
    print(f"[INFO] ViT model loaded  | threshold={vit_threshold:.3f} "
          f"| test_acc={checkpoint.get('test_accuracy', 0) * 100:.1f}% "
          f"| test_auc={checkpoint.get('test_auc', 0):.3f}")


try:
    load_vit()
except Exception as e:
    print(f"[ERROR] Could not load ViT checkpoint '{VIT_CHECKPOINT}': {e}")
    traceback.print_exc()
    model = None


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes):
    """Returns (tensor, pil_img_resized) ready for ViT inference."""
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    tensor  = transform(pil_img).unsqueeze(0).to(DEVICE)
    resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
    return tensor, resized


def encode_preview(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":       "ok",
        "model_type":   "vit",
        "model_loaded": model is not None,
        "threshold":    round(vit_threshold, 3),
        "device":       str(DEVICE),
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a field named 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
        return jsonify({"error": f"Unsupported type '{ext}'. Use JPG or PNG."}), 400

    try:
        image_bytes = file.read()
        tensor, pil_img = preprocess(image_bytes)

        # allow caller to override threshold; default to the val-tuned value
        threshold = float(request.form.get("threshold", vit_threshold))

        with torch.no_grad():
            logit = model(pixel_values=tensor).logits.squeeze()
            prob  = float(torch.sigmoid(logit).item())

        label      = "PNEUMONIA" if prob >= threshold else "NORMAL"
        confidence = prob if label == "PNEUMONIA" else (1.0 - prob)

        return jsonify({
            "label":       label,
            "probability": round(prob, 4),
            "confidence":  round(confidence * 100, 2),
            "threshold":   round(threshold, 3),
            "model_type":  "vit",
            "preview":     encode_preview(pil_img),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed. See server logs."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)