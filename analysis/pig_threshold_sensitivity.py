#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run the frozen PIG-only hysteresis-threshold sensitivity analysis.

Protocol
--------
- Input: an aligned continuous YOLO-EAAF probability GeoTIFF, binary
  ground-truth GeoTIFF, and binary evaluation-domain GeoTIFF.
- Otsu base threshold: estimated from probability values > 1e-4 inside
  the evaluation domain.
- High multipliers: 0.25 to 0.85 in steps of 0.05.
- Low multipliers: 0.05 to 0.45 in steps of 0.05.
- Only combinations with High > Low are retained (102 combinations).
- Post-processing for every combination:
    1. hysteresis thresholding;
    2. binary opening, disk radius 1;
    3. remove connected objects smaller than 350 pixels;
    4. binary dilation, disk radius 2.
- Selection objective: macro F1.
- Deterministic tie handling follows the original loop order:
  smaller High first, then smaller Low.
- Getz/TILE4 is not used.

Outputs
-------
- pig_threshold_sensitivity_expanded_grid.csv
- pig_threshold_sensitivity_summary.json
- pig_best_damage_mask.tif
- pig_threshold_sensitivity_heatmap.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from skimage import morphology
from skimage.filters import apply_hysteresis_threshold, threshold_otsu


DEFAULT_HIGH_VALUES = tuple(
    np.round(np.arange(0.25, 0.90, 0.05), 2).tolist()
)
DEFAULT_LOW_VALUES = tuple(
    np.round(np.arange(0.05, 0.50, 0.05), 2).tolist()
)

DEFAULT_MIN_PROBABILITY = 1e-4
DEFAULT_OPENING_RADIUS = 1
DEFAULT_MIN_OBJECT_SIZE = 350
DEFAULT_DILATION_RADIUS = 2

OUTPUT_NODATA = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen PIG-only hysteresis-threshold sensitivity "
            "analysis on aligned probability, reference, and domain rasters."
        )
    )
    parser.add_argument(
        "--probability-map",
        type=Path,
        required=True,
        help="Continuous YOLO-EAAF probability GeoTIFF.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Aligned binary ground-truth GeoTIFF.",
    )
    parser.add_argument(
        "--evaluation-domain",
        type=Path,
        required=True,
        help=(
            "Aligned binary evaluation-domain GeoTIFF. Pixels greater than "
            "zero are evaluated."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/output_pig_threshold_sensitivity"),
        help="Directory for CSV, JSON, best mask, and heatmap outputs.",
    )
    parser.add_argument(
        "--region",
        default="PIG",
        help="Region label written to output tables. Default: PIG.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=DEFAULT_MIN_PROBABILITY,
        help="Minimum probability included in Otsu estimation. Default: 1e-4.",
    )
    parser.add_argument(
        "--opening-radius",
        type=int,
        default=DEFAULT_OPENING_RADIUS,
        help="Binary-opening disk radius in pixels. Default: 1.",
    )
    parser.add_argument(
        "--min-object-size",
        type=int,
        default=DEFAULT_MIN_OBJECT_SIZE,
        help="Minimum retained object size in pixels. Default: 350.",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=DEFAULT_DILATION_RADIUS,
        help="Binary-dilation disk radius in pixels. Default: 2.",
    )
    parser.add_argument(
        "--skip-heatmap",
        action="store_true",
        help="Skip creation of the PNG heatmap.",
    )
    return parser.parse_args()


def validate_parameters(args: argparse.Namespace) -> None:
    if args.min_probability < 0:
        raise ValueError("--min-probability must be non-negative.")
    if args.opening_radius < 0:
        raise ValueError("--opening-radius must be non-negative.")
    if args.min_object_size < 0:
        raise ValueError("--min-object-size must be non-negative.")
    if args.dilation_radius < 0:
        raise ValueError("--dilation-radius must be non-negative.")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")


def is_finite_nodata(value: float | int | None) -> bool:
    return value is not None and bool(np.isfinite(value))


def read_raster(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile.copy()
        metadata = {
            "shape": array.shape,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }
    return array, profile, metadata


def validate_alignment(
    reference_metadata: dict[str, Any],
    comparison_metadata: dict[str, Any],
    label: str,
) -> None:
    if comparison_metadata["shape"] != reference_metadata["shape"]:
        raise ValueError(f"{label} shape does not match the probability map.")
    if comparison_metadata["crs"] != reference_metadata["crs"]:
        raise ValueError(f"{label} CRS does not match the probability map.")
    if not comparison_metadata["transform"].almost_equals(
        reference_metadata["transform"]
    ):
        raise ValueError(
            f"{label} transform does not match the probability map."
        )


def valid_data_mask(
    array: np.ndarray,
    nodata: float | int | None,
) -> np.ndarray:
    valid = np.isfinite(array)
    if is_finite_nodata(nodata):
        valid &= array != nodata
    return valid


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def calculate_metrics(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, int | float]:
    y_true = ground_truth.astype(bool)
    y_pred = prediction.astype(bool)

    tp = int(np.logical_and(y_pred, y_true).sum())
    fp = int(np.logical_and(y_pred, ~y_true).sum())
    fn = int(np.logical_and(~y_pred, y_true).sum())
    tn = int(np.logical_and(~y_pred, ~y_true).sum())

    damage_precision = safe_divide(tp, tp + fp)
    damage_recall = safe_divide(tp, tp + fn)
    damage_f1 = safe_divide(
        2.0 * damage_precision * damage_recall,
        damage_precision + damage_recall,
    )
    damage_iou = safe_divide(tp, tp + fp + fn)

    background_precision = safe_divide(tn, tn + fn)
    background_recall = safe_divide(tn, tn + fp)
    background_f1 = safe_divide(
        2.0 * background_precision * background_recall,
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

    return {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "damage_precision": float(damage_precision),
        "damage_recall": float(damage_recall),
        "damage_f1": float(damage_f1),
        "damage_iou": float(damage_iou),
        "background_precision": float(background_precision),
        "background_recall": float(background_recall),
        "background_f1": float(background_f1),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def postprocess_probability_map(
    probability_map: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    opening_radius: int,
    min_object_size: int,
    dilation_radius: int,
) -> np.ndarray:
    raw_binary_mask = apply_hysteresis_threshold(
        probability_map,
        low_threshold,
        high_threshold,
    ).astype(np.uint8)

    cleaned_mask = raw_binary_mask.astype(bool)

    if opening_radius > 0:
        cleaned_mask = morphology.binary_opening(
            cleaned_mask,
            footprint=morphology.disk(opening_radius),
        )

    if min_object_size > 0:
        cleaned_mask = morphology.remove_small_objects(
            cleaned_mask,
            min_size=min_object_size,
        )

    if dilation_radius > 0:
        cleaned_mask = morphology.binary_dilation(
            cleaned_mask,
            footprint=morphology.disk(dilation_radius),
        )

    return cleaned_mask.astype(np.uint8)


def write_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("No sensitivity rows were generated.")

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_best_mask(
    output_path: Path,
    best_mask: np.ndarray,
    evaluation_domain: np.ndarray,
    profile: dict[str, Any],
) -> None:
    output = np.full(
        best_mask.shape,
        OUTPUT_NODATA,
        dtype=np.uint8,
    )
    output[evaluation_domain] = 0
    output[evaluation_domain & best_mask.astype(bool)] = 1

    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=OUTPUT_NODATA,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(output, 1)


def create_heatmap(
    rows: list[dict[str, Any]],
    best_result: dict[str, Any],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as error:
        raise RuntimeError(
            "Matplotlib is required for heatmap output. "
            "Install matplotlib or use --skip-heatmap."
        ) from error

    high_values = list(DEFAULT_HIGH_VALUES)
    low_values = list(DEFAULT_LOW_VALUES)

    matrix = np.full(
        (len(high_values), len(low_values)),
        np.nan,
        dtype=np.float64,
    )

    high_index = {
        value: index
        for index, value in enumerate(high_values)
    }
    low_index = {
        value: index
        for index, value in enumerate(low_values)
    }

    for row in rows:
        matrix[
            high_index[round(float(row["high_mult"]), 2)],
            low_index[round(float(row["low_mult"]), 2)],
        ] = float(row["macro_f1"])

    display_matrix = matrix[::-1]
    display_high_values = high_values[::-1]

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(
        display_matrix,
        aspect="auto",
    )
    figure.colorbar(image, ax=axis, label="Macro F1")

    axis.set_xticks(np.arange(len(low_values)))
    axis.set_xticklabels(
        [f"{value:.2f}" for value in low_values],
        rotation=45,
        ha="right",
    )
    axis.set_yticks(np.arange(len(display_high_values)))
    axis.set_yticklabels(
        [f"{value:.2f}" for value in display_high_values]
    )
    axis.set_xlabel("Low Threshold Multiplier")
    axis.set_ylabel("High Threshold Multiplier")
    axis.set_title("PIG-only Hysteresis-Threshold Sensitivity")

    for row_index in range(display_matrix.shape[0]):
        for column_index in range(display_matrix.shape[1]):
            value = display_matrix[row_index, column_index]
            if np.isfinite(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    best_high = round(float(best_result["high_mult"]), 2)
    best_low = round(float(best_result["low_mult"]), 2)
    best_row = display_high_values.index(best_high)
    best_column = low_values.index(best_low)

    axis.add_patch(
        Rectangle(
            (best_column - 0.5, best_row - 0.5),
            1,
            1,
            fill=False,
            linewidth=2.5,
        )
    )
    axis.scatter(
        best_column,
        best_row,
        marker="*",
        s=180,
        facecolors="none",
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_parameters(args)

    require_file(args.probability_map, "probability map")
    require_file(args.ground_truth, "ground truth")
    require_file(args.evaluation_domain, "evaluation domain")

    probability, profile, probability_metadata = read_raster(
        args.probability_map
    )
    ground_truth, _, ground_truth_metadata = read_raster(
        args.ground_truth
    )
    evaluation_domain, _, evaluation_metadata = read_raster(
        args.evaluation_domain
    )

    validate_alignment(
        probability_metadata,
        ground_truth_metadata,
        "Ground truth",
    )
    validate_alignment(
        probability_metadata,
        evaluation_metadata,
        "Evaluation domain",
    )

    probability = probability.astype(np.float32)

    probability_valid = valid_data_mask(
        probability,
        probability_metadata["nodata"],
    )
    ground_truth_valid = valid_data_mask(
        ground_truth,
        ground_truth_metadata["nodata"],
    )
    evaluation_valid = valid_data_mask(
        evaluation_domain,
        evaluation_metadata["nodata"],
    )

    domain = (
        probability_valid
        & ground_truth_valid
        & evaluation_valid
        & (evaluation_domain > 0)
    )

    if not np.any(domain):
        raise RuntimeError("The evaluation domain is empty.")

    ground_truth_binary = (ground_truth > 0).astype(np.uint8)

    valid_probabilities = probability[
        domain & (probability > args.min_probability)
    ]
    if valid_probabilities.size == 0:
        raise RuntimeError(
            "No positive probabilities were found inside the evaluation "
            "domain for Otsu estimation."
        )

    otsu_base_threshold = float(
        threshold_otsu(valid_probabilities)
    )

    probability_for_processing = np.where(
        probability_valid,
        probability,
        0.0,
    ).astype(np.float32)

    threshold_pairs = [
        (high_multiplier, low_multiplier)
        for high_multiplier in DEFAULT_HIGH_VALUES
        for low_multiplier in DEFAULT_LOW_VALUES
        if high_multiplier > low_multiplier
    ]

    if len(threshold_pairs) != 102:
        raise RuntimeError(
            f"Expected 102 valid threshold combinations, "
            f"but found {len(threshold_pairs)}."
        )

    y_true = ground_truth_binary[domain].astype(np.uint8)

    rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_mask: np.ndarray | None = None

    for high_multiplier, low_multiplier in threshold_pairs:
        high_threshold = otsu_base_threshold * high_multiplier
        low_threshold = otsu_base_threshold * low_multiplier

        prediction_mask = postprocess_probability_map(
            probability_map=probability_for_processing,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            opening_radius=args.opening_radius,
            min_object_size=args.min_object_size,
            dilation_radius=args.dilation_radius,
        )

        y_predicted = prediction_mask[domain].astype(np.uint8)
        metrics = calculate_metrics(
            ground_truth=y_true,
            prediction=y_predicted,
        )

        row = {
            "ice_shelf": args.region,
            "high_mult": float(high_multiplier),
            "low_mult": float(low_multiplier),
            "high_threshold_absolute": float(high_threshold),
            "low_threshold_absolute": float(low_threshold),
            **metrics,
        }
        rows.append(row)

        # Exact original selection behavior:
        # update only when macro F1 is strictly greater.
        # The ascending loop order resolves exact ties as:
        # smaller High first, then smaller Low.
        if (
            best_result is None
            or row["macro_f1"] > best_result["macro_f1"]
        ):
            best_result = row.copy()
            best_mask = (
                prediction_mask.astype(bool) & domain
            ).astype(np.uint8)

    if best_result is None or best_mask is None:
        raise RuntimeError("The grid search produced no valid result.")

    high_on_boundary = bool(
        np.isclose(
            best_result["high_mult"],
            min(DEFAULT_HIGH_VALUES),
        )
        or np.isclose(
            best_result["high_mult"],
            max(DEFAULT_HIGH_VALUES),
        )
    )
    low_on_boundary = bool(
        np.isclose(
            best_result["low_mult"],
            min(DEFAULT_LOW_VALUES),
        )
        or np.isclose(
            best_result["low_mult"],
            max(DEFAULT_LOW_VALUES),
        )
    )

    best_result.update(
        {
            "selection_objective": "macro_f1",
            "tie_rule": (
                "Original ascending loop order; update only for strictly "
                "greater macro_f1."
            ),
            "high_on_boundary": high_on_boundary,
            "low_on_boundary": low_on_boundary,
            "boundary_optimum": (
                high_on_boundary or low_on_boundary
            ),
            "otsu_base_threshold": otsu_base_threshold,
            "getz_used": False,
        }
    )

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["damage_f1"]),
            -float(row["damage_iou"]),
            float(row["high_mult"]),
            float(row["low_mult"]),
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        args.output_dir
        / "pig_threshold_sensitivity_expanded_grid.csv"
    )
    summary_path = (
        args.output_dir
        / "pig_threshold_sensitivity_summary.json"
    )
    best_mask_path = (
        args.output_dir
        / "pig_best_damage_mask.tif"
    )
    heatmap_path = (
        args.output_dir
        / "pig_threshold_sensitivity_heatmap.png"
    )

    write_csv(csv_path, sorted_rows)
    write_best_mask(
        best_mask_path,
        best_mask,
        domain,
        profile,
    )

    summary = {
        "analysis_region": args.region,
        "getz_used": False,
        "selection_objective": "macro_f1",
        "tie_rule": (
            "Original ascending loop order; update only for strictly "
            "greater macro_f1."
        ),
        "high_search_values": list(DEFAULT_HIGH_VALUES),
        "low_search_values": list(DEFAULT_LOW_VALUES),
        "valid_combination_count": len(rows),
        "minimum_probability_for_otsu": args.min_probability,
        "opening_radius": args.opening_radius,
        "minimum_object_size": args.min_object_size,
        "dilation_radius": args.dilation_radius,
        "evaluation_pixels": int(domain.sum()),
        "ground_truth_damage_pixels": int(
            ground_truth_binary[domain].sum()
        ),
        "best_tested_configuration": best_result,
        "inputs": {
            "probability_map": str(args.probability_map),
            "ground_truth": str(args.ground_truth),
            "evaluation_domain": str(args.evaluation_domain),
        },
        "outputs": {
            "results_csv": str(csv_path),
            "best_mask": str(best_mask_path),
            "heatmap": (
                None
                if args.skip_heatmap
                else str(heatmap_path)
            ),
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    if not args.skip_heatmap:
        create_heatmap(
            rows=rows,
            best_result=best_result,
            output_path=heatmap_path,
        )

    print("PIG-only threshold sensitivity analysis completed.")
    print("Getz/TILE4 was not used.")
    print(f"Valid threshold combinations: {len(rows)}")
    print(f"Otsu base threshold: {otsu_base_threshold:.10f}")
    print(
        f"Best High: {best_result['high_mult']:.2f}"
    )
    print(
        f"Best Low: {best_result['low_mult']:.2f}"
    )
    print(
        f"Best Macro F1: {best_result['macro_f1']:.10f}"
    )
    print(
        f"Best Damage F1: {best_result['damage_f1']:.10f}"
    )
    print(
        f"Best Damage IoU: {best_result['damage_iou']:.10f}"
    )
    print(f"CSV output: {csv_path}")
    print(f"JSON output: {summary_path}")
    print(f"Best mask: {best_mask_path}")
    if not args.skip_heatmap:
        print(f"Heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
