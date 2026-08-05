#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate aligned binary 30 m rasters to a common 300 m grid and evaluate them.

Protocol reproduced from the manuscript analysis:
- 10 x 10 native 30 m pixels per 300 m cell.
- Primary occupancy threshold: 1% (any positive native pixel).
- Sensitivity thresholds: 5% and 10%.
- A 300 m cell is evaluated only when all 100 native pixels exist and
  all 100 belong to the supplied evaluation domain.
- The same occupancy threshold is applied to the ground truth and prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine


BLOCK_FACTOR = 10
SOURCE_RESOLUTION_M = 30
TARGET_RESOLUTION_M = 300
OCCUPANCY_THRESHOLDS = (0.01, 0.05, 0.10)
PRIMARY_OCCUPANCY_THRESHOLD = 0.01
MIN_SUPPORT_FRACTION = 1.0
MIN_EVALUATION_FRACTION = 1.0


@dataclass
class Grid:
    height: int
    width: int
    transform: Affine
    crs: Any
    profile: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate aligned binary 30 m prediction and reference rasters "
            "to a common 300 m grid and calculate evaluation metrics."
        )
    )
    parser.add_argument(
        "--region",
        default="region",
        help="Region label used in output filenames and tables.",
    )
    parser.add_argument(
        "--reference-grid",
        type=Path,
        required=True,
        help=(
            "A 30 m GeoTIFF defining the canonical grid, CRS, extent, "
            "and affine transform."
        ),
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Aligned binary 30 m ground-truth GeoTIFF.",
    )
    parser.add_argument(
        "--evaluation-domain",
        type=Path,
        required=True,
        help=(
            "Aligned binary 30 m evaluation-domain GeoTIFF. "
            "Pixels greater than zero belong to the evaluation domain."
        ),
    )
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "Prediction specification. Use NAME=PATH and repeat this option "
            "to evaluate multiple aligned 30 m predictions."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/output_300m"),
        help="Directory for 300 m rasters, CSV tables, and protocol JSON.",
    )
    return parser.parse_args()


def parse_prediction_specs(specs: list[str]) -> dict[str, Path]:
    predictions: dict[str, Path] = {}

    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid prediction specification: {spec!r}. "
                "Use NAME=PATH."
            )

        name, path_text = spec.split("=", 1)
        name = name.strip()
        path_text = path_text.strip()

        if not name:
            raise ValueError(
                f"Prediction name is empty in specification: {spec!r}"
            )
        if not path_text:
            raise ValueError(
                f"Prediction path is empty in specification: {spec!r}"
            )
        if name in predictions:
            raise ValueError(
                f"Duplicate prediction name: {name!r}"
            )

        predictions[name] = Path(path_text)

    return predictions


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_grid(path: Path) -> Grid:
    with rasterio.open(path) as src:
        return Grid(
            height=src.height,
            width=src.width,
            transform=src.transform,
            crs=src.crs,
            profile=src.profile.copy(),
        )


def assert_aligned(
    path: Path,
    reference: Grid,
    label: str,
) -> None:
    with rasterio.open(path) as src:
        if (
            src.height != reference.height
            or src.width != reference.width
            or src.transform != reference.transform
            or src.crs != reference.crs
        ):
            raise RuntimeError(
                f"{label} is not aligned with the 30 m reference grid: {path}"
            )


def validate_inputs(
    reference_grid_path: Path,
    ground_truth_path: Path,
    evaluation_domain_path: Path,
    predictions: dict[str, Path],
) -> Grid:
    require_file(reference_grid_path, "reference grid")
    require_file(ground_truth_path, "ground truth")
    require_file(evaluation_domain_path, "evaluation domain")

    for model_name, path in predictions.items():
        require_file(path, f"{model_name} prediction")

    reference = read_grid(reference_grid_path)

    if not (
        np.isclose(abs(reference.transform.a), SOURCE_RESOLUTION_M)
        and np.isclose(abs(reference.transform.e), SOURCE_RESOLUTION_M)
    ):
        raise RuntimeError(
            "The reference-grid pixel size is not 30 m."
        )

    assert_aligned(
        ground_truth_path,
        reference,
        "Ground truth",
    )
    assert_aligned(
        evaluation_domain_path,
        reference,
        "Evaluation domain",
    )

    for model_name, path in predictions.items():
        assert_aligned(
            path,
            reference,
            f"{model_name} prediction",
        )

    return reference


def read_binary(path: Path) -> np.ndarray:
    """
    Reproduce the original binary conversion rule:
    every finite value greater than zero is treated as class 1.
    """
    with rasterio.open(path) as src:
        array = src.read(1)

    return (
        np.isfinite(array) & (array > 0)
    ).astype(np.uint8)


def make_300m_grid(reference: Grid) -> Grid:
    height = math.ceil(reference.height / BLOCK_FACTOR)
    width = math.ceil(reference.width / BLOCK_FACTOR)
    transform = reference.transform * Affine.scale(
        BLOCK_FACTOR,
        BLOCK_FACTOR,
    )

    profile = reference.profile.copy()
    profile.update(
        height=height,
        width=width,
        transform=transform,
        count=1,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )

    return Grid(
        height=height,
        width=width,
        transform=transform,
        crs=reference.crs,
        profile=profile,
    )


def block_fraction(
    binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return positive-pixel occupancy and native-pixel support fractions
    for each 10 x 10 block.
    """
    height, width = binary.shape
    padded_height = (
        math.ceil(height / BLOCK_FACTOR) * BLOCK_FACTOR
    )
    padded_width = (
        math.ceil(width / BLOCK_FACTOR) * BLOCK_FACTOR
    )

    padded = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32,
    )
    padded[:height, :width] = binary.astype(np.float32)

    support = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32,
    )
    support[:height, :width] = 1.0

    fraction = padded.reshape(
        padded_height // BLOCK_FACTOR,
        BLOCK_FACTOR,
        padded_width // BLOCK_FACTOR,
        BLOCK_FACTOR,
    ).mean(axis=(1, 3))

    support_fraction = support.reshape(
        padded_height // BLOCK_FACTOR,
        BLOCK_FACTOR,
        padded_width // BLOCK_FACTOR,
        BLOCK_FACTOR,
    ).mean(axis=(1, 3))

    return (
        fraction.astype(np.float32),
        support_fraction.astype(np.float32),
    )


def save_raster(
    path: Path,
    array: np.ndarray,
    grid: Grid,
    dtype: str,
    nodata: int | float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    profile = grid.profile.copy()
    profile.update(
        dtype=dtype,
        nodata=nodata,
    )

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)


def calculate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    evaluation: np.ndarray,
) -> dict[str, int | float]:
    """
    Reproduce the confusion-matrix and metric definitions used
    in the original 300 m evaluation script.
    """
    y_true = ground_truth[evaluation].astype(bool)
    y_pred = prediction[evaluation].astype(bool)

    tp = int(np.logical_and(y_pred, y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())
    eps = 1e-12

    damage_precision = tp / (tp + fp + eps)
    damage_recall = tp / (tp + fn + eps)
    damage_f1 = (
        2
        * damage_precision
        * damage_recall
        / (damage_precision + damage_recall + eps)
    )
    damage_iou = tp / (tp + fp + fn + eps)

    background_precision = tn / (tn + fn + eps)
    background_recall = tn / (tn + fp + eps)
    background_f1 = (
        2
        * background_precision
        * background_recall
        / (background_precision + background_recall + eps)
    )

    return {
        "macro_precision": float(
            (damage_precision + background_precision) / 2
        ),
        "macro_recall": float(
            (damage_recall + background_recall) / 2
        ),
        "macro_f1": float(
            (damage_f1 + background_f1) / 2
        ),
        "damage_precision": float(damage_precision),
        "damage_recall": float(damage_recall),
        "damage_f1": float(damage_f1),
        "damage_iou": float(damage_iou),
        "background_precision": float(background_precision),
        "background_recall": float(background_recall),
        "background_f1": float(background_f1),
        "accuracy": float(
            (tp + tn) / (tp + fp + fn + tn + eps)
        ),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "evaluation_pixels": int(evaluation.sum()),
        "damage_pixels": int(y_true.sum()),
        "predicted_damage_pixels": int(y_pred.sum()),
    }


def occupancy_name(threshold: float) -> str:
    return (
        f"occ_{int(round(threshold * 100)):02d}pct"
    )


def safe_filename(text: str) -> str:
    cleaned = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text.strip(),
    )
    return cleaned.strip("_") or "item"


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError(
            f"No rows are available for CSV output: {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    predictions = parse_prediction_specs(
        args.prediction
    )

    reference = validate_inputs(
        reference_grid_path=args.reference_grid,
        ground_truth_path=args.ground_truth,
        evaluation_domain_path=args.evaluation_domain,
        predictions=predictions,
    )
    grid_300m = make_300m_grid(reference)

    region_name = args.region.strip() or "region"
    region_slug = safe_filename(region_name)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_30m = read_binary(
        args.ground_truth
    )
    evaluation_domain_30m = read_binary(
        args.evaluation_domain
    )

    (
        ground_truth_fraction,
        support_fraction,
    ) = block_fraction(ground_truth_30m)

    (
        evaluation_fraction,
        support_check,
    ) = block_fraction(evaluation_domain_30m)

    if not np.allclose(
        support_fraction,
        support_check,
    ):
        raise RuntimeError(
            "Inconsistent support fractions were produced."
        )

    common_evaluation = (
        (support_fraction >= MIN_SUPPORT_FRACTION)
        & (
            evaluation_fraction
            >= MIN_EVALUATION_FRACTION
        )
    )

    if not np.any(common_evaluation):
        raise RuntimeError(
            "The common 300 m evaluation domain is empty."
        )

    print("=" * 82)
    print("COMMON 300 m EVALUATION")
    print(f"Region: {region_name}")
    print(
        f"30 m shape: {reference.height} x "
        f"{reference.width}"
    )
    print(
        f"300 m shape: {grid_300m.height} x "
        f"{grid_300m.width}"
    )
    print(
        "Primary rule: 1% occupancy / any-positive"
    )
    print("Sensitivity rules: 5% and 10% occupancy")
    print(
        "Evaluation cells require 100% native-pixel "
        "support and 100% evaluation-domain coverage."
    )
    print(
        f"Common evaluation cells: "
        f"{int(common_evaluation.sum())}"
    )
    print("=" * 82)

    save_raster(
        output_dir
        / f"{region_slug}_support_fraction_300m.tif",
        support_fraction,
        grid_300m,
        "float32",
        0.0,
    )
    save_raster(
        output_dir
        / f"{region_slug}_evaluation_fraction_300m.tif",
        evaluation_fraction,
        grid_300m,
        "float32",
        0.0,
    )
    save_raster(
        output_dir
        / f"{region_slug}_evaluation_domain_300m.tif",
        common_evaluation.astype(np.uint8),
        grid_300m,
        "uint8",
        0,
    )
    save_raster(
        output_dir
        / f"{region_slug}_ground_truth_occupancy_300m.tif",
        ground_truth_fraction,
        grid_300m,
        "float32",
        0.0,
    )

    prediction_fractions: dict[str, np.ndarray] = {}

    for model_name, path in predictions.items():
        fraction, prediction_support = block_fraction(
            read_binary(path)
        )

        if not np.allclose(
            prediction_support,
            support_fraction,
        ):
            raise RuntimeError(
                f"{model_name}: inconsistent support fraction."
            )

        prediction_fractions[model_name] = fraction

        save_raster(
            output_dir
            / (
                f"{region_slug}_"
                f"{safe_filename(model_name)}_"
                "occupancy_300m.tif"
            ),
            fraction,
            grid_300m,
            "float32",
            0.0,
        )

    rows: list[dict[str, Any]] = []

    for threshold in OCCUPANCY_THRESHOLDS:
        label = occupancy_name(threshold)
        threshold_dir = output_dir / label
        threshold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        ground_truth_300m = (
            ground_truth_fraction >= threshold
        ).astype(np.uint8)
        ground_truth_300m[~common_evaluation] = 0

        save_raster(
            threshold_dir
            / (
                f"{region_slug}_ground_truth_"
                f"{label}_300m.tif"
            ),
            ground_truth_300m,
            grid_300m,
            "uint8",
            0,
        )

        print(
            f"\n[{region_name}] occupancy threshold "
            f"= {threshold:.0%}"
        )
        print(
            f"  Evaluation cells: "
            f"{int(common_evaluation.sum())}"
        )
        print(
            f"  Damage cells: "
            f"{int(ground_truth_300m[common_evaluation].sum())}"
        )
        print(
            f"  Damage fraction: "
            f"{float(ground_truth_300m[common_evaluation].mean()):.6f}"
        )

        for model_name, fraction in prediction_fractions.items():
            prediction_300m = (
                fraction >= threshold
            ).astype(np.uint8)
            prediction_300m[~common_evaluation] = 0

            save_raster(
                threshold_dir
                / (
                    f"{region_slug}_"
                    f"{safe_filename(model_name)}_"
                    f"{label}_300m.tif"
                ),
                prediction_300m,
                grid_300m,
                "uint8",
                0,
            )

            result = calculate_metrics(
                ground_truth_300m,
                prediction_300m,
                common_evaluation,
            )

            row = {
                "region": region_name,
                "model": model_name,
                "resolution_m": TARGET_RESOLUTION_M,
                "source_resolution_m": SOURCE_RESOLUTION_M,
                "aggregation": "occupancy threshold",
                "occupancy_threshold": float(threshold),
                "primary_experiment": bool(
                    np.isclose(
                        threshold,
                        PRIMARY_OCCUPANCY_THRESHOLD,
                    )
                ),
                "support_fraction_required": (
                    MIN_SUPPORT_FRACTION
                ),
                "evaluation_fraction_required": (
                    MIN_EVALUATION_FRACTION
                ),
                **result,
            }
            rows.append(row)

            print(
                f"  {model_name} | "
                f"Macro F1={result['macro_f1']:.6f} | "
                f"Damage F1={result['damage_f1']:.6f} | "
                f"IoU={result['damage_iou']:.6f}"
            )

    all_metrics_path = (
        output_dir
        / f"{region_slug}_all_300m_metrics.csv"
    )
    write_csv(
        all_metrics_path,
        rows,
    )

    primary_rows = [
        row
        for row in rows
        if row["primary_experiment"]
    ]
    primary_rows.sort(
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["damage_f1"]),
            str(row["model"]),
        )
    )

    primary_metrics_path = (
        output_dir
        / f"{region_slug}_primary_1pct_300m_metrics.csv"
    )
    write_csv(
        primary_metrics_path,
        primary_rows,
    )

    summary = {
        "region": region_name,
        "target_resolution_m": TARGET_RESOLUTION_M,
        "source_resolution_m": SOURCE_RESOLUTION_M,
        "block_factor": BLOCK_FACTOR,
        "primary_occupancy_threshold": (
            PRIMARY_OCCUPANCY_THRESHOLD
        ),
        "sensitivity_occupancy_thresholds": list(
            OCCUPANCY_THRESHOLDS
        ),
        "minimum_support_fraction": (
            MIN_SUPPORT_FRACTION
        ),
        "minimum_evaluation_fraction": (
            MIN_EVALUATION_FRACTION
        ),
        "ground_truth_rule": (
            "Same occupancy threshold as model outputs."
        ),
        "prediction_models": list(predictions.keys()),
        "primary_results": primary_rows,
    }

    summary_path = (
        output_dir
        / f"{region_slug}_300m_protocol_and_results.json"
    )
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 82)
    print("COMMON 300 m EVALUATION COMPLETED")
    print(f"All metrics: {all_metrics_path}")
    print(f"Primary metrics: {primary_metrics_path}")
    print(f"Protocol summary: {summary_path}")
    print("=" * 82)


if __name__ == "__main__":
    main()
