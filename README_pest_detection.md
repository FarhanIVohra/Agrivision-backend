# Pest Detection API — Deployment Guide (Render-ready)

This document explains how to deploy and run the Pest Detection ML flow with CPU-only inference and a stable API contract.

## Overview
- Endpoint: `POST /api/pest-detection`
- Input: multipart/form-data, single file param `image` (JPEG/PNG)
- Output JSON:
  {
    "status": "success",
    "disease": "<prediction>",
    "confidence": <float 0-1>,
    "severity": "low|medium|high",
    "treatment": "<recommendation>",
    "notes": "Upload clearer image if confidence is low."
  }
- Fallback rule: if `confidence < 0.45` ->
  - `disease: "Unknown"`
  - `treatment: "Unable to identify. Please upload clearer image."`

## Model File
- Place the model under: `backend/models/pest_detection_model.h5`
  - Alternatively supported names (fallback): `backend/models/pest_disease_model.h5`
  - The service loads the model via a RELATIVE path; no absolute paths are used.
- Recommended lightweight options:
  - MobileNetV2 (PlantVillage checkpoint) exported to `.h5`
  - YOLOv8n ONNX or PyTorch if needed (adjust service accordingly)

## Backend (FastAPI)
- Location: `backend/app`
- Server: `uvicorn app.main:app --port 8000 --reload`
- CORS configured for local dev and typical ports
- Model loads once at startup via dependency injection in `pest_detection.py`
- CPU-only inference enforced (TensorFlow configured to avoid GPU reliance)

### Health/Info Endpoints
- `GET /api/pest-detection/model-info` — returns model path and metadata
- `GET /api/pest-detection/supported-diseases` — returns supported disease list

## Frontend (Vite React)
- Location: `frontend`
- Dev server: `npm run start -- --port 5173`
- UI Page: `src/pages/pest-detection`
  - Upload component: `components/ImageUploadArea.jsx` (uses `FormData` with `image`)
  - Results component: `components/AnalysisResults.jsx` (renders disease, confidence, severity, treatment)

## Deployment on Render
1. Create a new Render Web Service (Docker or Native environment).
2. Set build command for backend service:
   - `pip install -r backend/requirements.txt`
3. Start command:
   - `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Ensure model file is included in repo at `backend/models/pest_detection_model.h5` (keep the file small).
5. Use CPU-only; no CUDA/GPU dependencies required.
6. Expose the service publicly; path is `/api/pest-detection`.

## Sample Responses

### 1) Confident result
```
{
  "status": "success",
  "disease": "Leaf Spot",
  "confidence": 0.87,
  "severity": "high",
  "treatment": "Use Copper-based fungicide and ensure proper sunlight",
  "notes": "Upload clearer image if confidence is low."
}
```

### 2) Low confidence fallback
```
{
  "status": "success",
  "disease": "Unknown",
  "confidence": 0.32,
  "severity": "low",
  "treatment": "Unable to identify. Please upload clearer image.",
  "notes": "Upload clearer image if confidence is low."
}
```

## Quick Local Test
- Start backend: `cd backend && python -m uvicorn app.main:app --port 8000 --reload`
- Start frontend: `cd frontend && npm run start -- --port 5173`
- Open `http://localhost:5173/` → go to Pest Detection → upload a test image.

## Notes
- If the model file is missing, the service returns a safe response with low confidence.
- The backend returns structured JSON errors for missing/invalid images and inference failures.
- You can enrich `treatment_map` in `pest_detection.py` to cover more diseases.