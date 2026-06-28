---
title: PneumoScanAI-Backend
emoji: 🫁
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PneumoScanAI - Deep Learning Backend API

This repository hosts the Flask-based backend application for **PneumoScanAI**. It serves a fine-tuned Vision Transformer (ViT) model (`google/vit-base-patch16-224-in21k`) trained to classify chest X-ray images for pneumonia detection.

The application is containerized with Docker and optimized to deploy seamlessly on Hugging Face Spaces using their high-RAM free CPU infrastructure.

## 🚀 System Architecture

- **Framework:** Flask with Flask-CORS enabled for secure cross-origin resource sharing.
- **Deep Learning Layer:** PyTorch & Hugging Face Transformers (`ViTForImageClassification`).
- **Container Environment:** Docker (`python:3.10-slim` baseline image running on port `7860`).

## 🛠️ API Endpoints

### 1. Health Check
Checks the status of the server and verifies if the model weights are fully initialized.
- **URL:** `/health`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "ok",
    "model_type": "vit",
    "model_loaded": true,
    "threshold": 0.5,
    "device": "cpu"
  }