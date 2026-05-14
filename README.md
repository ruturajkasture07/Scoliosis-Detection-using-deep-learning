# ScolioVis - Deep Learning Scoliosis Detection

This project implements the automated pipeline for detecting vertebrae, calculating Cobb angles, and allowing doctors/researchers to analyze different model outputs.

## Project Structure
- `frontend/` - Next.js 14 Web Application (React, TypeScript, Tailwind CSS, Zustand)
- `backend/` - Python FastAPI Application (PyTorch, OpenCV, Math Engine)
- `Models/` - Storage for the model weights (`.pth`, `.pt`)

## 1. Backend Setup

The backend handles the Keypoint Detection Model inference and the Cobb Angle mathematical computation.

1. Open a terminal in `backend/`
2. Create a virtual environment (optional but recommended): `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows)
4. Install requirements: `pip install -r requirements.txt`
5. Run the server: `uvicorn main:app --reload`
   The backend API will start at `http://localhost:8000`.

*Note: The current `main.py` has a mock inference function that works without loading the heavy model weights immediately, but the infrastructure to load the exact `.pth` and `.pt` models is present.*

## 2. Frontend Setup

The frontend provides the premium UI for model selection and visual analysis.

1. Open a terminal in `frontend/`
2. Install dependencies (already installed): `npm install`
3. Run the development server: `npm run dev`
4. Open your browser and navigate to `http://localhost:3000`

## Features Implemented
- **Premium UI:** Dark-mode glassmorphism aesthetic with responsive canvas.
- **Model Selection:** Users can choose between Keypoint R-CNN, YOLOv8 Pose, VerteNet, and Residual-Unet to perform comparative analysis.
- **5-Step Visualization:**
  1. Blue bounding boxes for gross vertebra detection.
  2. Cyan 4-corner keypoints (dots).
  3. Solid white horizontal mid-lines for tilt estimation.
  4. Math engine correctly calculates the maximum tilt and determines End Vertebrae.
  5. Intersecting colored lines drawn to represent PT (Orange), MT (Magenta), and TL (Green) curves.

## Integration Note
To use the actual models, modify `backend/main.py` to replace `mock_inference` with your real PyTorch model prediction code. The frontend is fully hooked up and ready to receive the standard JSON response format.
