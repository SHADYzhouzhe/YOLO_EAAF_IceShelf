# YOLO-EAAF: Deep Learning-Guided Energy-Augmented Anisotropic Frangi Filtering for Antarctic Ice-Shelf Damage Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21794050.svg)](https://doi.org/10.5281/zenodo.21794050)

This repository provides the public implementation of **YOLO-EAAF**, a workflow that combines YOLO oriented-bounding-box localization with energy-augmented anisotropic Frangi filtering for Antarctic ice-shelf surface-damage detection.

The repository includes code for:

- YOLO-OBB training;
- generation of continuous YOLO-EAAF probability maps;
- conversion of probability maps to binary damage masks;
- 30 m pixel-level evaluation;
- 30 m to 300 m occupancy aggregation and evaluation;
- PIG-only hysteresis-threshold sensitivity analysis.

The associated derived data are archived on Zenodo:

**DOI:** [10.5281/zenodo.21794050](https://doi.org/10.5281/zenodo.21794050)

## Repository structure

```text
YOLO_EAAF_IceShelf/
|-- analysis/
|   `-- pig_threshold_sensitivity.py
|-- configs/
|   `-- glacier_example.yaml
|-- evaluation/
|   |-- aggregate_to_300m.py
|   `-- evaluate_30m.py
|-- examples/
|   |-- expected_output/
|   |   `-- README.md
|   `-- input/
|       `-- test_sample_PIG.tif
|-- figures/
|   `-- PIG_damage_time_series_1999_2020.png
|-- inference/
|   |-- generate_damage_mask.py
|   `-- generate_probability_map.py
|-- training/
|   `-- train_yolo_obb.py
|-- weights/
|   `-- README.md
|-- CITATION.cff
|-- LICENSE
|-- README.md
`-- requirements.txt
```

## Environment

The public release was checked with Python 3.11 and the versions listed in `requirements.txt`.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned requirements install a CPU-compatible PyTorch build from the default Python package index. For GPU training or large-area inference, install a PyTorch build compatible with the local CUDA environment before installing the remaining dependencies.

## Model checkpoint

The trained checkpoint is not committed to the Git repository.

Obtain the `best.pt` checkpoint from the GitHub Release asset for version `v1.0.0` and save it as:

```text
weights/best.pt
```

The inference script expects this path by default.

## Training

Edit the dataset root in:

```text
configs/glacier_example.yaml
```

The example configuration follows the Ultralytics YOLO-OBB dataset layout:

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Run training:

```bash
python training/train_yolo_obb.py
```

The public training script uses:

```text
pretrained model: yolo26s-obb.pt
epochs: 300
image size: 512
batch size: 16
workers: 4
```

Training outputs are written under:

```text
YOLO_EAAF_Training/
```

## Generate a continuous probability map

The default example uses:

```text
input:  examples/input/test_sample_PIG.tif
weight: weights/best.pt
output: examples/output/test_sample_probability.tif
```

Run:

```bash
python inference/generate_probability_map.py
```

The script applies the following frozen inference settings:

```text
window size: 512 pixels
stride: 256 pixels
ROI padding: 40 pixels
YOLO confidence threshold: 0.45
filter method: anisotropic
```

The output is a single-band Float32 GeoTIFF on the input raster grid.

To process another raster, edit the `TILE_TASKS` list near the top of:

```text
inference/generate_probability_map.py
```

## Convert the probability map to a binary damage mask

Run with the default example paths:

```bash
python inference/generate_damage_mask.py
```

For exact manuscript-style threshold estimation, provide the aligned evaluation-domain raster used for the analysis:

```bash
python inference/generate_damage_mask.py \
  --input path/to/probability_map.tif \
  --evaluation-domain path/to/evaluation_domain.tif \
  --output path/to/damage_mask.tif
```

The frozen default post-processing workflow is:

```text
Otsu sample: valid probability values > 1e-4
high multiplier: 0.35
low multiplier: 0.30
binary opening radius: 1 pixel
minimum object size: 350 pixels
binary dilation radius: 2 pixels
```

Output encoding:

```text
0   background
1   damage
255 nodata / outside retained domain
```

## 30 m evaluation

Evaluate an aligned prediction and reference raster:

```bash
python evaluation/evaluate_30m.py \
  --prediction path/to/prediction_30m.tif \
  --reference path/to/reference_30m.tif
```

Optionally restrict the comparison to a valid domain:

```bash
python evaluation/evaluate_30m.py \
  --prediction path/to/prediction_30m.tif \
  --reference path/to/reference_30m.tif \
  --valid-mask path/to/evaluation_domain_30m.tif
```

Default outputs:

```text
evaluation/metrics_30m.json
evaluation/metrics_30m.csv
```

The script checks raster shape, CRS, and affine transform before evaluation.

## 300 m aggregation and evaluation

The 300 m script aggregates aligned 30 m binary rasters using 10 × 10 pixel blocks. It evaluates occupancy thresholds of 1%, 5%, and 10%; 1% is the primary threshold.

Example:

```bash
python evaluation/aggregate_to_300m.py \
  --region Getz \
  --reference-grid path/to/reference_grid_30m.tif \
  --ground-truth path/to/ground_truth_30m.tif \
  --evaluation-domain path/to/evaluation_domain_30m.tif \
  --prediction YOLO-EAAF=path/to/yolo_eaaf_30m.tif \
  --prediction U-Net=path/to/unet_30m.tif \
  --prediction SegFormer=path/to/segformer_30m.tif \
  --output-dir evaluation/output_300m
```

The same occupancy threshold is applied to both reference and prediction. Only 300 m cells with complete support and complete evaluation-domain coverage are retained.

The script writes:

- 300 m occupancy rasters;
- support and common-domain rasters;
- thresholded binary rasters;
- full metrics CSV;
- primary-threshold metrics CSV;
- protocol and result JSON files.

## PIG threshold-sensitivity analysis

The public sensitivity script consumes precomputed, aligned rasters:

- continuous YOLO-EAAF probability map;
- binary ground truth;
- binary evaluation domain.

Run:

```bash
python analysis/pig_threshold_sensitivity.py \
  --probability-map path/to/PIG_probability_map.tif \
  --ground-truth path/to/PIG_ground_truth.tif \
  --evaluation-domain path/to/PIG_evaluation_domain.tif \
  --output-dir analysis/output_pig_threshold_sensitivity
```

Frozen sensitivity protocol:

```text
region: PIG only
Getz/TILE4 used for selection: no
high multipliers: 0.25–0.85, step 0.05
low multipliers: 0.05–0.45, step 0.05
constraint: High > Low
valid combinations: 102
selection objective: macro F1
```

Each threshold pair uses the same morphology as the final mask-generation script:

```text
binary opening radius: 1 pixel
minimum object size: 350 pixels
binary dilation radius: 2 pixels
```

The original ascending loop order is retained. For exactly equal macro-F1 values, the first encountered combination is retained, corresponding to the smaller High multiplier and then the smaller Low multiplier.

Default outputs:

```text
pig_threshold_sensitivity_expanded_grid.csv
pig_threshold_sensitivity_summary.json
pig_best_damage_mask.tif
pig_threshold_sensitivity_heatmap.png
```

## PIG damage time-series figure

The repository retains the study figure:

```text
figures/PIG_damage_time_series_1999_2020.png
```

This figure is included for documentation and visualization. The annual time-series source products are not distributed in the public Zenodo package.

## Data availability

Derived data supporting the study are available from Zenodo:

[https://doi.org/10.5281/zenodo.21794050](https://doi.org/10.5281/zenodo.21794050)

The Zenodo data package and this GitHub software repository are separate release components:

- **software:** MIT License;
- **Zenodo data:** CC BY 4.0.

The trained model checkpoint is distributed separately as a GitHub Release asset and is not tracked in the Git repository.

## Citation

GitHub can generate citation metadata from `CITATION.cff`.

Software citation metadata:

```text
Zhou, Z., An, L., Li, J., & Yang, Z. (2026).
Deep Learning-Guided Energy-Augmented Anisotropic Frangi Filtering
for Antarctic Ice-Shelf Damage Detection. Version 1.0.0.
https://doi.org/10.5281/zenodo.21794050
```

When citing the associated journal article, use the final bibliographic information assigned by the journal after publication.

## License

The source code in this repository is released under the [MIT License](LICENSE).

The associated Zenodo data package is released under the Creative Commons Attribution 4.0 International license.
