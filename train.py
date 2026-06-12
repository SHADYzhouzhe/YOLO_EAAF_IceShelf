"""
YOLO-OBB Training Pipeline for Antarctic Ice Shelf Damage Detection
Repository: YOLO-EAAF-IceShelf
"""
import os
from ultralytics import YOLO

# ==========================================
# Training Configuration
# ==========================================
# Ensure these paths are configured in your environment or passed via args
MODEL_PRETRAINED = 'yolo26s-obb.pt'
DATASET_CONFIG = 'glacier.yaml'  # Ensure this file exists in your repo
PROJECT_NAME = 'YOLO_EAAF_Training'
NAME = 'train_obb_optim'

def train_yolo_obb():
    # Load model
    model = YOLO(MODEL_PRETRAINED)

    # Begin training
    model.train(
        data=DATASET_CONFIG,
        epochs=300,
        imgsz=512,
        batch=16,
        workers=4,
        project=PROJECT_NAME,
        name=NAME,

        # Data Augmentation (Rigidly restricted for geometric fidelity)
        degrees=90.0,       # Random rotation for arbitrary crevasse orientation
        shear=0.0,          # Prohibited to preserve orthogonal projection
        perspective=0.0,    # Prohibited to prevent artificial distortion
        fliplr=0.5,
        flipud=0.5,

        # Optimization Strategy
        lr0=0.005,
        patience=50,
        mosaic=1.0,
        mixup=0.2
    )

if __name__ == '__main__':
    # Add simple check to ensure data config exists
    if os.path.exists(DATASET_CONFIG):
        train_yolo_obb()
    else:
        print(f"Error: {DATASET_CONFIG} not found. Please verify the path.")