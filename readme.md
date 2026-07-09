# 🫁 PneumoScanAI

A full-stack web application for **chest X-ray pneumonia detection** powered by a fine-tuned **ConvNeXt-Tiny** CNN. Upload a single X-ray or an entire folder of scans, get instant predictions with confidence scores, and explore model metrics — all through a polished React UI backed by a Flask API.

> **Disclaimer:** This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. All results must be reviewed by a qualified radiologist or physician.

---

## 📸 Features

| Tab | What it does |
|---|---|
| 🔬 **Single Predict** | Drag-and-drop or browse a single chest X-ray → instant label, confidence meter, probability bar |
| 📂 **Batch Predict** | Upload a whole folder of X-rays → live progress bar, per-image results table, 4 interactive charts (bar, pie, trend line, confidence histogram), CSV export |
| 📈 **Model Metrics** | Real training curves (loss / accuracy / AUC), Phase 1→2 boundary marker, confusion matrix, CNN architecture diagram |
| ℹ️ **About** | Dataset summary, end-to-end pipeline walkthrough, tech stack |

**Other UI details:**
- 🌙 / ☀️ Dark & Light mode toggle with full colour contrast in both themes
- Sticky glassmorphism header
- Animated drop zones, scan-line overlay during inference
- Threshold slider (0.05 – 0.95, step 0.005) with one-click reset to validated optimum
- Export batch results as CSV

---

## 🧠 Model

| Property | Value |
|---|---|
| Architecture | **ConvNeXt-Tiny** (torchvision, ImageNet-1k pretrained) |
| Total parameters | 28,033,505 |
| Input size | 224 × 224 px |
| Normalisation | ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]` |
| Loss | `BCEWithLogitsLoss` with `pos_weight=0.346` (handles 2.89:1 class imbalance) |
| Optimiser | AdamW + Cosine Annealing (Phase 1) / Cosine Warmup (Phase 2) |
| Training strategy | 2-phase fine-tuning (see below) |
| Best epoch | 7 — val AUC **0.9926**, train/val gap **1.8%** ✓ |
| Test Accuracy | **90%** @ threshold 0.5 |
| Test ROC-AUC | **0.9587** |
| Val-tuned threshold | **0.092** (max-F1 on validation set — catches 99.2% of pneumonia cases) |
| Checkpoint file | `pneumonia_cnn_checkpoint.pt` |

### 2-Phase Fine-Tuning Strategy

**Phase 1 — Head warm-up (5 epochs)**
- ConvNeXt backbone fully frozen
- Only the 3-layer MLP classification head trained (214,913 params)
- Learning rate: `1e-3`, scheduler: `CosineAnnealingLR`

**Phase 2 — Partial backbone fine-tuning (2 epochs)**
- Backbone frozen except `features.6` + `features.7` (last downsampling + final stage)
- Backbone LR: `5e-6` (tiny, to preserve pretrained features)
- Head LR: `1e-4`
- Scheduler: cosine with warmup

**Custom Classification Head** (replaces the original `Linear(768→1000)`):
```
Linear(768 → 256) → GELU → Dropout(0.4)
Linear(256 → 64)  → GELU → Dropout(0.3)
Linear(64  → 1)   → Sigmoid (applied in loss / inference)
```

### Confusion Matrix (test set, 624 images)

**At threshold = 0.5** (best balanced accuracy):

|  | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** | TN = 189 | FP = 45 |
| **Actual PNEUMONIA** | FN = 19 | TP = 371 |

- Accuracy: **90%** · Precision: 89% · Recall: 95% · F1: 92%

**At threshold = 0.092** (val-tuned, max recall):

|  | Predicted NORMAL | Predicted PNEUMONIA |
|---|---|---|
| **Actual NORMAL** | TN = 147 | FP = 87 |
| **Actual PNEUMONIA** | FN = 3 | TP = 387 |

- Accuracy: 86% · Recall: **99.2%** (only 3 missed pneumonia cases)

> Use threshold **0.5** for balanced screening, **0.092** when missing a pneumonia case is clinically unacceptable.

---

## 📁 Project Structure

```
pneumonia_app/
│
├── README.md
│
├── backend/
│   ├── app.py                        ← Flask API (ConvNeXt inference)
│   ├── requirements.txt              ← pip dependencies
│   └── pneumonia_cnn_checkpoint.pt   ← ← paste your trained checkpoint here
│
├── frontend/
│   ├── package.json
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── index.js
│       └── App.jsx                   ← full React UI (4 tabs, dark/light mode)
│
└── training/
    ├── pneumonia_cnn.py              ← ConvNeXt-Tiny training script (local)
    └── pneumonia_vit.py              ← ViT-Base training script (reference)
```

---

## 🗂️ Dataset

**Chest X-Ray Images (Pneumonia)** by Paul Mooney — [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,341 | 3,875 | 5,216 |
| Test | 234 | 390 | 624 |
| Val | 8 | 8 | 16 |

> **Note:** The 16-image `val/` folder is too small for reliable metrics. The training script carves a stratified 15% validation split from `train/` instead (~783 images), leaving `test/` as the final unbiased holdout — used **once**, at the very end.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- (Optional but recommended) NVIDIA GPU with CUDA for faster inference

### Step 1 — Clone / download the project

```bash
git clone https://github.com/your-username/pneumoscan-ai.git
cd pneumoscan-ai/pneumonia_app
```

### Step 2 — Copy your trained checkpoint

Download `pneumonia_cnn_checkpoint.pt` from Kaggle's notebook Output tab after training, then:

```
pneumonia_app/backend/pneumonia_cnn_checkpoint.pt   ← paste here
```

### Step 3 — Start the Flask backend

```bash
cd backend

# Create & activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# Install dependencies
pip install -r requirements.txt

# Start the API server
python app.py
# → Running on http://localhost:5000
```

Verify it's working:
```bash
curl http://localhost:5000/health
# {"device":"cpu","model_loaded":true,"model_type":"convnext_tiny","status":"ok","threshold":0.092}
```

### Step 4 — Start the React frontend

Open a second terminal:

```bash
cd frontend
npm install
npm start
# → Opens http://localhost:3000
```

---

## 🌐 API Reference

### `GET /health`

Returns model status.

```json
{
  "status": "ok",
  "model_type": "convnext_tiny",
  "model_loaded": true,
  "threshold": 0.092,
  "device": "cuda"
}
```

### `POST /predict`

**Form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | image file | ✅ | JPEG, PNG, BMP, or TIFF chest X-ray |
| `threshold` | float | ❌ | Override decision threshold (default: val-tuned value from checkpoint) |

**Response:**

```json
{
  "label":       "PNEUMONIA",
  "probability": 0.9946,
  "confidence":  99.46,
  "threshold":   0.092,
  "model_type":  "convnext_tiny",
  "preview":     "data:image/jpeg;base64,..."
}
```

**Error responses:**

| Status | Meaning |
|---|---|
| 400 | No file, empty filename, or unsupported extension |
| 500 | Inference failed (check server logs) |
| 503 | Checkpoint not loaded (missing `.pt` file) |

---

## 🏋️ Training on Kaggle

The training notebook (`pneumonia-detection-cnn.ipynb`) is designed to run on Kaggle with a free GPU:

1. Go to [kaggle.com](https://www.kaggle.com) → **Create Notebook**
2. **Add Input** → search `chest-xray-pneumonia` (paultimothymooney) → Add
3. Enable **GPU T4 x2** under Settings → Accelerator
4. Upload the notebook and **Run All** (`Ctrl+F9`)
5. After completion, download `pneumonia_cnn_checkpoint.pt` from the **Output** tab

**Expected training time:** ~15 minutes on T4 GPU (5 + 2 epochs)

---

## 🔧 Threshold Guide

The default threshold of **0.092** was chosen to maximise F1 on the validation set. Because this is a medical screening tool, missing a real pneumonia case (false negative) is generally worse than a false alarm (false positive).

| Threshold | Use case | Accuracy | Recall | FPR |
|---|---|---|---|---|
| **0.092** | Clinical screening — minimise missed pneumonia | 86% | **99.2%** | 37% |
| **0.5** | Balanced — best overall accuracy | **90%** | 95% | 19% |
| **0.7+** | High specificity — minimise false alarms | ~87% | ~85% | ~8% |

You can adjust the threshold in real time using the slider in the UI, or pass it as a form field in the API request.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch 2.3 + TorchVision 0.18 (ConvNeXt-Tiny) |
| Training | Kaggle GPU notebook (Tesla T4) |
| Backend API | Flask 3.0 + Flask-CORS |
| Image handling | Pillow |
| Frontend | React 18 + Recharts |
| Styling | Inline CSS with dark/light theme tokens |

---

## 📦 Dependencies

**Backend (`requirements.txt`):**
```
flask==3.0.3
flask-cors==4.0.1
Pillow==10.3.0
numpy==1.26.4
torch==2.3.0
torchvision==0.18.0
```

**Frontend (`package.json`):**
```
react ^18.3.1
react-dom ^18.3.1
react-scripts 5.0.1
recharts ^2.12.7
```

---

## ⚠️ Important Notes

- **This is not a medical device.** Never use predictions from this tool as the sole basis for a clinical decision.
- The model was trained and evaluated on the Kaggle Chest X-Ray Pneumonia dataset. Performance on images from different scanners, hospitals, or patient populations may vary.
- The val-tuned threshold (0.092) is very sensitive by design. At this threshold, ~37% of normal X-rays are flagged as pneumonia — a radiologist must review all flagged cases.
- The training/validation gap is only **1.8%**, indicating the model generalises well to unseen data.

---

## 👤 Author

Built as an end-to-end medical imaging deep learning project — from dataset exploration and model training on Kaggle, to a production-ready Flask + React web application with dark/light mode, batch prediction, and interactive analytics.
