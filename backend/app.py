"""
app.py — Flask backend for PneumoScanAI
Serves the fine-tuned ConvNeXt-Tiny model:
  pneumonia_cnn_checkpoint.pt

Place that checkpoint file in this same backend/ folder before starting.
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
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

app = Flask(__name__)
CORS(app)

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT = os.environ.get("CNN_CHECKPOINT", "pneumonia_cnn_checkpoint.pt")

model         = None
threshold     = 0.7    # val-tuned F1-optimal; overwritten from checkpoint
IMG_SIZE      = 224
# ImageNet stats used by ConvNeXt-Tiny (pulled from torchvision weight metadata)
IMG_MEAN      = [0.485, 0.456, 0.406]
IMG_STD       = [0.229, 0.224, 0.225]
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Loader ────────────────────────────────────────────────────────────────────
def load_model():
    global model, threshold, IMG_SIZE, IMG_MEAN, IMG_STD

    print(f"[INFO] Device  : {DEVICE}")
    print(f"[INFO] Loading checkpoint: {CHECKPOINT}")

    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)

    # Read preprocessing params saved at training time
    threshold = float(ckpt.get("threshold", 0.7))
    IMG_SIZE  = int(ckpt.get("img_size",  224))
    IMG_MEAN  = list(ckpt.get("img_mean", [0.485, 0.456, 0.406]))
    IMG_STD   = list(ckpt.get("img_std",  [0.229, 0.224, 0.225]))

    # Rebuild the exact same architecture used in training
    # convnext_tiny backbone with a custom 3-layer MLP head replacing the
    # original Linear(768→1000) classification head.
    net = convnext_tiny(weights=None)   # no pretrained weights; we load our own
    in_features = net.classifier[2].in_features   # 768 for convnext_tiny

    net.classifier[2] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(0.4),
        nn.Linear(256, 64),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )

    net.load_state_dict(ckpt["model_state_dict"])
    net.to(DEVICE)
    net.eval()

    model = net
    print(f"[INFO] ConvNeXt-Tiny loaded  "
          f"| threshold={threshold:.3f} "
          f"| test_acc={ckpt.get('test_accuracy', 0)*100:.1f}% "
          f"| test_auc={ckpt.get('test_auc', 0):.4f}")


try:
    load_model()
except Exception as e:
    print(f"[ERROR] Could not load checkpoint '{CHECKPOINT}': {e}")
    traceback.print_exc()
    model = None


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image_bytes):
    """Returns (tensor, pil_img_resized) ready for ConvNeXt inference."""
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
        "model_type":   "convnext_tiny",
        "model_loaded": model is not None,
        "threshold":    round(threshold, 4),
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

        # Caller can override threshold; default is the val-tuned value from checkpoint
        thr = float(request.form.get("threshold", threshold))

        with torch.no_grad():
            logit = model(tensor).squeeze()
            prob  = float(torch.sigmoid(logit).item())

        label      = "PNEUMONIA" if prob >= thr else "NORMAL"
        confidence = prob if label == "PNEUMONIA" else (1.0 - prob)

        return jsonify({
            "label":       label,
            "probability": round(prob, 4),
            "confidence":  round(confidence * 100, 2),
            "threshold":   round(thr, 4),
            "model_type":  "convnext_tiny",
            "preview":     encode_preview(pil_img),
        })

    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed. See server logs."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)