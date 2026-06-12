# YOLO-EAAF: Deep Learning-Guided Energy-Augmented Anisotropic Frangi Filter for Antarctic Ice Shelf Damage Detection

This repository contains the official implementation of the **YOLO-EAAF** framework, a physics-guided deep learning approach for high-precision detection of ice shelf surface damage (e.g., micro-crevasses, rifts, and mélange) on Antarctic ice shelves.

## Overview
The YOLO-EAAF framework decouples macroscopic region localization from microscopic feature extraction. By integrating the **YOLOv26-OBB** model for rapid ROI localization and the **Energy-Augmented Anisotropic Frangi (EAAF)** filter for physical feature filtering, our method overcomes the interpretability limitations and high manual annotation costs associated with standalone deep learning models.

## Key Features
- **Physics-Guided:** Integrates Hessian-based Frangi filtering with anisotropic diffusion to preserve authentic crevasse edges.
- **High Fidelity:** Operates at native image resolution, overcoming the sub-pixel blurring common in traditional models.
- **Robust Generalization:** Demonstrated cross-sensor robustness across Sentinel-1 (SAR), Sentinel-2, and Landsat 8 (optical) imagery.

## Prerequisites
The framework requires the following core dependencies (see `requirements.txt` for full list):
- `ultralytics` (YOLO-OBB)
- `rasterio` & `rioxarray` (Geospatial processing)
- `scikit-image` & `scipy` (EAAF physical filtering)
- `torch` & `tensorflow`

Install all dependencies via:
```bash
pip install -r requirements.txt