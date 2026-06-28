# PneumoScanAi 🫁🔍

PneumoScanAi is a production-ready, full-stack medical imaging application designed to assist healthcare professionals in detecting pneumonia from chest X-ray images. The project leverages a state-of-the-art **Vision Transformer (ViT)** model for high-accuracy classification, served via a lightweight **Flask REST API**, and wrapped in a responsive **React.js** web interface.

---

## 🚀 Key Features

* **Advanced ViT Architecture:** Uses a fine-tuned Vision Transformer optimized specifically for complex medical image nuances.
* **Real-Time Classification:** Instant image processing and inference pipeline providing quick diagnostic support.
* **Granular Confidence Metrics:** Displays classification status (Normal vs. Pneumonia) along with precise probability scores.
* **Meticulous Experiment Tracking:** Model training metrics, loss curves, and hyperparameter tuning tracked seamlessly using Azure Machine Learning.
* **Modern Dashboard:** Clean, minimal, and responsive user interface designed for intuitive interaction.

---

## 📊 Model Performance & Evaluation

The deep learning model was trained, validated, and strictly evaluated on an independent test split of chest X-ray imagery. 

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Test Accuracy** | `87.34%` | Overall correctness on unseen patient data |
| **Area Under Curve (AUC)** | `0.9452` | High discriminative ability with minimal false positives |
| **Classification Threshold** | `0.773` | Tuned optimized threshold balancing sensitivity and precision |

---

## 🛠️ Tech Stack

### Frontend
* **React.js** (Functional Components, Hooks)
* **Axios** (Asynchronous API state handling)
* **Tailwind CSS / Styled Components** (For a clean medical dashboard UI)

### Backend
* **Flask** (Python RESTful API)
* **PyTorch & Torchvision** (Model loading, tensor transformations, and image normalization)
* **Hugging Face Transformers** (Vision Transformer backbone architecture)

### DevOps & Cloud MLOps
* **Azure Machine Learning** (Environment management, artifact versioning, and run tracking)

---

## ⚙️ Project Architecture & Data Flow

1. **Upload:** The user uploads a standard chest X-ray image via the React frontend dashboard.
2. **Ingestion & Preprocessing:** The image is transmitted via a `POST` request to the Flask backend, where it is automatically resized, cropped, and normalized to match the strict tensor inputs required by the ViT.
3. **Inference:** The processed tensor is passed through the fine-tuned ViT. The raw logits are squashed into a probability distribution.
4. **Decision Boundary:** The system applies the optimized threshold of $0.773$ to accurately bucket the output into `Normal` or `Pneumonia`.
5. **Payload Delivery:** The UI receives the calculated probability and final label, updating dynamically in milliseconds.

---

## 💻 Installation & Local Setup

### Prerequisites
* Python 3.8+
* Node.js (v16 or higher) & npm

### 1. Backend Configuration (Flask)
```bash
# Navigate to the backend folder
cd backend

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows: venv\Scripts\activate
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt

# Run the development server
python app.py

# Navigate to the frontend folder
cd ../frontend

# Install node packages
npm install

# Start the local development web server
npm start

🤝 Acknowledgments & References
Hugging Face for the robust Vision Transformer implementations.

Azure ML for providing enterprise-grade infrastructure to track and log deep learning workloads.

Medical open-source communities for providing labeled chest X-ray datasets.