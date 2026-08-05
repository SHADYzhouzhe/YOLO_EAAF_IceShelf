#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate a binary ice-shelf damage prediction at the native grid.

The script reports:
- TP, FP, FN, and TN
- damage-class precision, recall, F1, and IoU
- background-class precision, recall, and F1
- macro precision, macro recall, and macro F1
- overall accuracy

Prediction, reference, and optional valid-domain mask must share
the same raster grid.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a binary damage prediction on the native "
            "30 m raster grid."
        )
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="Predicted binary damage-mask GeoTIFF.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference binary damage-mask GeoTIFF.",
    )
    parser.add_argument(
        "--valid-mask",
        type=Path,
        default=None,
        help=(
            "Optional valid evaluation-domain mask. "
            "Pixels greater than zero are evaluated."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("evaluation/metrics_30m.json"),
        help="Output JSON file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("evaluation/metrics_30m.csv"),
        help="Output CSV file.",
    )
    return parser.parse_args()


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def read_raster(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Raster not found: {path}"
        )

    with rasterio.open(path) as src:
        array = src.read(1)
        metadata = {
            "shape": array.shape,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }

    return array, metadata


def validate_alignment(
    reference_meta: dict[str, Any],
    comparison_meta: dict[str, Any],
    comparison_name: str,
) -> None:
    if reference_meta["shape"] != comparison_meta["shape"]:
        raise ValueError(
            f"{comparison_name} shape does not match the "
            "reference raster."
        )

    if reference_meta["crs"] != comparison_meta["crs"]:
        raise ValueError(
            f"{comparison_name} CRS does not match the "
            "reference raster."
        )

    if not reference_meta["transform"].almost_equals(
        comparison_meta["transform"]
    ):
        raise ValueError(
            f"{comparison_name} transform does not match the "
            "reference raster."
        )


def build_valid_mask(
    prediction: np.ndarray,
    prediction_nodata: float | int | None,
    reference: np.ndarray,
    reference_nodata: float | int | None,
    domain_mask: np.ndarray | None,
    domain_nodata: float | int | None,
) -> np.ndarray:
    valid = np.isfinite(prediction)
    valid &= np.isfinite(reference)

    if prediction_nodata is not None:
        if np.isfinite(prediction_nodata):
            valid &= prediction != prediction_nodata

    if reference_nodata is not None:
        if np.isfinite(reference_nodata):
            valid &= reference != reference_nodata

    if domain_mask is not None:
        domain_valid = np.isfinite(domain_mask)

        if domain_nodata is not None:
            if np.isfinite(domain_nodata):
                domain_valid &= domain_mask != domain_nodata

        domain_valid &= domain_mask > 0
        valid &= domain_valid

    return valid


def calculate_metrics(
    prediction_binary: np.ndarray,
    reference_binary: np.ndarray,
) -> dict[str, int | float]:
    tp = int(
        np.count_nonzero(
            prediction_binary & reference_binary
        )
    )
    fp = int(
        np.count_nonzero(
            prediction_binary & ~reference_binary
        )
    )
    fn = int(
        np.count_nonzero(
            ~prediction_binary & reference_binary
        )
    )
    tn = int(
        np.count_nonzero(
            ~prediction_binary & ~reference_binary
        )
    )

    total = tp + fp + fn + tn

    damage_precision = safe_divide(
        tp,
        tp + fp,
    )
    damage_recall = safe_divide(
        tp,
        tp + fn,
    )
    damage_f1 = safe_divide(
        2 * damage_precision * damage_recall,
        damage_precision + damage_recall,
    )
    damage_iou = safe_divide(
        tp,
        tp + fp + fn,
    )

    background_precision = safe_divide(
        tn,
        tn + fn,
    )
    background_recall = safe_divide(
        tn,
        tn + fp,
    )
    background_f1 = safe_divide(
        2 * background_precision * background_recall,
        background_precision + background_recall,
    )

    macro_precision = (
        damage_precision + background_precision
    ) / 2.0
    macro_recall = (
        damage_recall + background_recall
    ) / 2.0
    macro_f1 = (
        damage_f1 + background_f1
    ) / 2.0

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "evaluated_pixels": total,
        "damage_precision": damage_precision,
        "damage_recall": damage_recall,
        "damage_f1": damage_f1,
        "damage_iou": damage_iou,
        "background_precision": background_precision,
        "background_recall": background_recall,
        "background_f1": background_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
    }


def write_json(
    output_path: Path,
    results: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )


def write_csv(
    output_path: Path,
    results: dict[str, Any],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results.keys()),
        )
        writer.writeheader()
        writer.writerow(results)


def main() -> None:
    args = parse_args()

    prediction, prediction_meta = read_raster(
        args.prediction
    )
    reference, reference_meta = read_raster(
        args.reference
    )

    validate_alignment(
        reference_meta,
        prediction_meta,
        "Prediction",
    )

    domain_mask = None
    domain_meta = None

    if args.valid_mask is not None:
        domain_mask, domain_meta = read_raster(
            args.valid_mask
        )
        validate_alignment(
            reference_meta,
            domain_meta,
            "Valid-domain mask",
        )

    valid = build_valid_mask(
        prediction=prediction,
        prediction_nodata=prediction_meta["nodata"],
        reference=reference,
        reference_nodata=reference_meta["nodata"],
        domain_mask=domain_mask,
        domain_nodata=(
            None
            if domain_meta is None
            else domain_meta["nodata"]
        ),
    )

    if not np.any(valid):
        raise RuntimeError(
            "No valid evaluation pixels were found."
        )

    prediction_binary = prediction[valid] > 0
    reference_binary = reference[valid] > 0

    metrics = calculate_metrics(
        prediction_binary,
        reference_binary,
    )

    results = {
        "prediction": str(args.prediction),
        "reference": str(args.reference),
        "valid_mask": (
            None
            if args.valid_mask is None
            else str(args.valid_mask)
        ),
        "evaluation_resolution_m": 30,
        **metrics,
    }

    write_json(
        args.output_json,
        results,
    )
    write_csv(
        args.output_csv,
        results,
    )

    print("30 m evaluation completed successfully.")
    print(f"Evaluated pixels: {metrics['evaluated_pixels']}")
    print(f"TP: {metrics['TP']}")
    print(f"FP: {metrics['FP']}")
    print(f"FN: {metrics['FN']}")
    print(f"TN: {metrics['TN']}")
    print(
        f"Damage precision: "
        f"{metrics['damage_precision']:.10f}"
    )
    print(
        f"Damage recall: "
        f"{metrics['damage_recall']:.10f}"
    )
    print(
        f"Damage F1: "
        f"{metrics['damage_f1']:.10f}"
    )
    print(
        f"Damage IoU: "
        f"{metrics['damage_iou']:.10f}"
    )
    print(
        f"Macro F1: "
        f"{metrics['macro_f1']:.10f}"
    )
    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.10f}"
    )
    print(f"JSON output: {args.output_json}")
    print(f"CSV output:  {args.output_csv}")


if __name__ == "__main__":
    main()
