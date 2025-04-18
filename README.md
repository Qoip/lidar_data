# Truck Bed Plane Detection

This project detects the bottom plane of truck beds from 3D LiDAR images. It includes a batch-processing pipeline and an interactive Jupyter notebook for exploring results.

## Features
- Plane detection using multiple methods
- Export of results to JSON (including detected plane and inlier points)
- Interactive visualization with 3D plots and overlays
- Jupyter UI for selecting files, categories, and methods

## Structure
- `main.py`: Processes all images in a directory and saves JSON results
- `notebooks/new.ipynb`: Interactive interface for exploring results with 3D plots
- `data/`: Folder with the dataset:
  - `data/images/` – contains images grouped by category
  - `data/ground_truth/` – contains corresponding polygon annotations

## Setup
```bash
pip install -r requirements.txt
```

## Usage
Run batch processing:
```bash
python main.py
```

Run the notebook:
```bash
jupyter notebook notebooks/new.ipynb
```

## Output Format
Each result is saved as JSON with:
- `plane_params`: Detected plane coefficients [A, B, C, D]
- `inlier_points`: Bottom hull points on the plane
- `filename`, `category`
