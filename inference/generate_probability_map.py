import os
import cv2
import numpy as np
import rasterio
from tqdm import tqdm
import warnings
from ultralytics import YOLO

# Feature extraction libraries
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")

# ==========================================
# 1. Global Configuration
# ==========================================
# Use relative paths for open-source repository standardization
YOLO_MODEL_PATH = "./weights/best.pt"

TILE_TASKS = [
    {
        "name": "PIG_Landsat8_Test",
        "input": "./examples/input/test_sample_PIG.tif",
        "output": "./examples/output/test_sample_probability.tif"
    }
]

# Algorithm Hyperparameters
IMG_SIZE = 512
STRIDE = 256
PADDING = 40
CONF_THRES = 0.45
FILTER_METHOD = 'anisotropic'
# ==========================================

# Initialize YOLO-OBB model
model = YOLO(YOLO_MODEL_PATH)


# ==========================================
# 2. Core EAAF Algorithm Library
# ==========================================
def perona_malik_diffusion(img, n_iter=15, kappa=0.05, gamma=0.15):
    """Anisotropic diffusion for edge-preserving noise reduction."""
    img_out = img.astype(np.float32).copy()
    safe_kappa = kappa + 1e-10
    for _ in range(n_iter):
        dn = img_out[:-2, 1:-1] - img_out[1:-1, 1:-1]
        ds = img_out[2:, 1:-1] - img_out[1:-1, 1:-1]
        de = img_out[1:-1, 2:] - img_out[1:-1, 1:-1]
        dw = img_out[1:-1, :-2] - img_out[1:-1, 1:-1]

        cn = np.exp(-(np.abs(dn) / safe_kappa) ** 2)
        cs = np.exp(-(np.abs(ds) / safe_kappa) ** 2)
        ce = np.exp(-(np.abs(de) / safe_kappa) ** 2)
        cw = np.exp(-(np.abs(dw) / safe_kappa) ** 2)

        img_out[1:-1, 1:-1] += gamma * (cn * dn + cs * ds + ce * de + cw * dw)
    return img_out


def guided_filter(I, p, radius, eps):
    I, p = I.astype(np.float32), p.astype(np.float32)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(p, cv2.CV_32F, (radius, radius))
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (radius, radius))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    return mean_a * I + mean_b


def compute_frangi_pipeline(image_preprocessed, sigmas=[1.0, 2.0, 4.0, 8.0]):
    """Multi-scale Hessian matrix computation."""
    frangi_max = np.zeros_like(image_preprocessed)
    beta, c = 0.5, 0.5
    for sigma in sigmas:
        H_elems = hessian_matrix(image_preprocessed, sigma=sigma, order='rc')
        l1, l2 = hessian_matrix_eigvals(H_elems)
        sort_idx = np.abs(l1) > np.abs(l2)
        l1_final = np.where(sort_idx, l2, l1)
        l2_final = np.where(sort_idx, l1, l2)
        Rb = (l1_final / (l2_final + 1e-10)) ** 2
        S2 = l1_final ** 2 + l2_final ** 2
        measure_linear = np.exp(-Rb / (2 * beta ** 2))
        measure_structure = (1 - np.exp(-S2 / (2 * c ** 2)))
        resp = np.nan_to_num(measure_linear * measure_structure)
        frangi_max = np.maximum(frangi_max, resp)
    return frangi_max


def apply_frangi_to_roi_float(roi_img, method='anisotropic'):
    """Energy-Augmented Anisotropic Frangi (EAAF) filter implementation."""
    if method == 'anisotropic':
        denoised = perona_malik_diffusion(roi_img)
    elif method == 'guided':
        denoised = guided_filter(roi_img, roi_img, 4, 1e-4)
    else:
        denoised = roi_img

    img_prep = 1.0 - denoised
    frangi_out = compute_frangi_pipeline(img_prep)

    # Local variance energy term calculation
    min_dim = min(img_prep.shape)
    u_size = 15 if min_dim > 15 else (min_dim // 2 * 2 + 1)
    mean_img = uniform_filter(img_prep, size=u_size)
    sq_mean_img = uniform_filter(img_prep ** 2, size=u_size)
    variance = np.clip(sq_mean_img - mean_img ** 2, 0, None)
    local_energy = np.sqrt(variance)

    # Hybrid response fusion
    hybrid = frangi_out + 0.05 * local_energy
    return np.nan_to_num(hybrid).astype(np.float32)


# ==========================================
# 3. Processing Pipeline
# ==========================================
def process_single_tile(tile_info):
    input_tif = tile_info["input"]
    output_tif = tile_info["output"]
    tile_name = tile_info["name"]

    print(f"\n" + "=" * 50)
    print(f"Initializing processing for: {tile_name}")
    print("=" * 50)

    if not os.path.exists(input_tif):
        print(f"Error: Input file not found: {input_tif}")
        print("Please check the relative path.")
        return

    # Read image
    with rasterio.open(input_tif) as src:
        raw_image = src.read(1)
        profile = src.profile
        height, width = raw_image.shape

    valid_mask = raw_image > 0
    if not np.any(valid_mask):
        print(f"[{tile_name}] No valid pixels found. Skipping.")
        return

    # Intensity Smoothing
    p2 = np.percentile(raw_image[valid_mask], 2)
    p98 = np.percentile(raw_image[valid_mask], 98)
    img_8u = np.clip(raw_image, p2, p98)
    img_8u = ((img_8u - p2) / (p98 - p2) * 255.0).astype(np.uint8)

    # Global CLAHE Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32, 32))
    img_global_enhanced = clahe.apply(img_8u)

    global_prob_map = np.zeros((height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    x_coords = list(range(0, width, STRIDE))
    y_coords = list(range(0, height, STRIDE))

    print(f">>> [PHASE 1] Sliding Window OBB Inference & EAAF Extraction ({tile_name})...")
    with tqdm(total=len(x_coords) * len(y_coords), desc=f"Processing {tile_name}") as pbar:
        for y in y_coords:
            for x in x_coords:
                x_off, y_off = x, y
                if x_off + IMG_SIZE > width: x_off = width - IMG_SIZE
                if y_off + IMG_SIZE > height: y_off = height - IMG_SIZE
                if x_off < 0 or y_off < 0:
                    pbar.update(1)
                    continue

                chip_enhanced = img_global_enhanced[y_off:y_off + IMG_SIZE, x_off:x_off + IMG_SIZE]
                if np.sum(chip_enhanced > 0) / chip_enhanced.size < 0.1:
                    pbar.update(1)
                    continue

                chip_bgr = np.repeat(chip_enhanced[:, :, np.newaxis], 3, axis=2)

                # YOLO-OBB Inference
                results = model.predict(chip_bgr, conf=CONF_THRES, augment=False, verbose=False)

                if len(results[0].obb) > 0:
                    obb_boxes_local = results[0].obb.xyxyxyxy.cpu().numpy()
                    for obb in obb_boxes_local:
                        # Coordinate transformation
                        obb_global = obb + np.array([x_off, y_off])
                        gx1, gy1 = int(np.min(obb_global[:, 0])), int(np.min(obb_global[:, 1]))
                        gx2, gy2 = int(np.max(obb_global[:, 0])), int(np.max(obb_global[:, 1]))

                        gx1_pad, gy1_pad = max(0, gx1 - PADDING), max(0, gy1 - PADDING)
                        gx2_pad, gy2_pad = min(width, gx2 + PADDING), min(height, gy2 + PADDING)

                        roi_enhanced_crop = img_global_enhanced[gy1_pad:gy2_pad, gx1_pad:gx2_pad]
                        roi_float = roi_enhanced_crop.astype(np.float32) / 255.0

                        if roi_float.size > 0:
                            # Apply EAAF Filter
                            roi_prob = apply_frangi_to_roi_float(roi_float, method=FILTER_METHOD)

                            obb_roi_coords = obb_global - np.array([gx1_pad, gy1_pad])
                            obb_roi_coords = np.round(obb_roi_coords).astype(np.int32)
                            obb_mask = np.zeros(roi_prob.shape, dtype=np.float32)
                            cv2.fillPoly(obb_mask, [obb_roi_coords], 1.0)

                            # Soft edge blending
                            # Note: Replaced skimage morphology.disk with cv2 getStructuringElement
                            kernel_blur = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                            obb_mask = cv2.dilate(obb_mask, kernel_blur, iterations=1)
                            obb_mask = cv2.GaussianBlur(obb_mask, (31, 31), 0)

                            local_mean = np.mean(roi_prob)
                            roi_prob_clean = np.clip(roi_prob - local_mean * 0.5, 0, None)
                            roi_prob_fused = roi_prob_clean * obb_mask

                            # Weighted Mosaic Fusion
                            global_prob_map[gy1_pad:gy2_pad, gx1_pad:gx2_pad] += roi_prob_fused
                            weight_map[gy1_pad:gy2_pad, gx1_pad:gx2_pad] += obb_mask
                pbar.update(1)

    print(f"\n>>> [PHASE 2] Mosaic Fusion & Exporting Probability Map ({tile_name})...")
    # Normalize overlapped regions
    safe_weight = np.clip(weight_map, 1e-6, None)
    global_prob_map = np.where(weight_map > 0, global_prob_map / safe_weight, 0)

    # Clip probabilities to strict [0, 1] range
    global_prob_map = np.clip(global_prob_map, 0.0, 1.0)

    # Mask out native nodata regions
    global_prob_map = np.where(valid_mask, global_prob_map, np.nan)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)

    # Export continuous probability map as Float32
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=np.nan)
    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(global_prob_map.astype(np.float32), 1)

    print(f"Success! Continuous probability map saved to: {output_tif}")


if __name__ == "__main__":
    for tile_info in TILE_TASKS:
        process_single_tile(tile_info)
