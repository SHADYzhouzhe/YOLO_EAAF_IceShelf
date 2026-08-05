#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert a continuous YOLO-EAAF probability map to a binary damage mask.

The default post-processing settings reproduce the frozen PIG threshold
sensitivity workflow inside the supplied evaluation domain:

1. Estimate an Otsu base threshold from valid probabilities > 1e-4.
2. Apply hysteresis thresholding with:
   high = Otsu * 0.35
   low  = Otsu * 0.30
3. Binary opening with a disk radius of 1 pixel.
4. Remove connected objects smaller than 350 pixels.
5. Binary dilation with a disk radius of 2 pixels.

For exact manuscript-style threshold estimation, provide the same
evaluation-domain raster used in the sensitivity analysis. Pixels outside
the valid/evaluation domain are written as nodata=255; valid output pixels
are encoded as 0 (background) and 1 (damage).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from skimage import morphology
from skimage.filters import apply_hysteresis_threshold, threshold_otsu


DEFAULT_INPUT = Path("examples/output/test_sample_probability.tif")
DEFAULT_OUTPUT = Path("examples/output/test_sample_damage_mask.tif")

DEFAULT_HIGH_MULTIPLIER = 0.35
DEFAULT_LOW_MULTIPLIER = 0.30
DEFAULT_MIN_PROBABILITY = 1e-4

DEFAULT_OPENING_RADIUS = 1
DEFAULT_MIN_OBJECT_SIZE = 350
DEFAULT_DILATION_RADIUS = 2

OUTPUT_NODATA = 255


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a continuous YOLO-EAAF probability map to a binary "
            "damage mask using the frozen hysteresis and morphology pipeline."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input continuous probability GeoTIFF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output binary damage-mask GeoTIFF.",
    )
    parser.add_argument(
        "--evaluation-domain",
        type=Path,
        default=None,
        help=(
            "Optional aligned evaluation-domain GeoTIFF. Pixels greater than "
            "zero are used for Otsu estimation and retained in the output."
        ),
    )
    parser.add_argument(
        "--high-multiplier",
        type=float,
        default=DEFAULT_HIGH_MULTIPLIER,
        help="High hysteresis-threshold multiplier. Default: 0.35.",
    )
    parser.add_argument(
        "--low-multiplier",
        type=float,
        default=DEFAULT_LOW_MULTIPLIER,
        help="Low hysteresis-threshold multiplier. Default: 0.30.",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=DEFAULT_MIN_PROBABILITY,
        help=(
            "Minimum positive probability included in Otsu estimation. "
            "Default: 1e-4."
        ),
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
        help="Minimum retained connected-object size in pixels. Default: 350.",
    )
    parser.add_argument(
        "--dilation-radius",
        type=int,
        default=DEFAULT_DILATION_RADIUS,
        help="Binary-dilation disk radius in pixels. Default: 2.",
    )
    return parser.parse_args()


def validate_parameters(args: argparse.Namespace) -> None:
    if args.high_multiplier <= 0:
        raise ValueError("--high-multiplier must be greater than zero.")
    if args.low_multiplier <= 0:
        raise ValueError("--low-multiplier must be greater than zero.")
    if args.high_multiplier <= args.low_multiplier:
        raise ValueError(
            "--high-multiplier must be greater than --low-multiplier."
        )
    if args.min_probability < 0:
        raise ValueError("--min-probability must be non-negative.")
    if args.opening_radius < 0:
        raise ValueError("--opening-radius must be non-negative.")
    if args.min_object_size < 0:
        raise ValueError("--min-object-size must be non-negative.")
    if args.dilation_radius < 0:
        raise ValueError("--dilation-radius must be non-negative.")


def is_finite_nodata(value: float | int | None) -> bool:
    return value is not None and bool(np.isfinite(value))


def read_probability(
    path: Path,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Probability raster not found: {path}")

    with rasterio.open(path) as src:
        probability = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        metadata = {
            "shape": probability.shape,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }

    return probability, profile, metadata


def read_evaluation_domain(
    path: Path,
    reference_metadata: dict[str, Any],
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation-domain raster not found: {path}")

    with rasterio.open(path) as src:
        domain = src.read(1)
        domain_metadata = {
            "shape": domain.shape,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }

    if domain_metadata["shape"] != reference_metadata["shape"]:
        raise ValueError(
            "Evaluation-domain shape does not match the probability raster."
        )
    if domain_metadata["crs"] != reference_metadata["crs"]:
        raise ValueError(
            "Evaluation-domain CRS does not match the probability raster."
        )
    if not domain_metadata["transform"].almost_equals(
        reference_metadata["transform"]
    ):
        raise ValueError(
            "Evaluation-domain transform does not match the probability raster."
        )

    domain_valid = np.isfinite(domain)
    if is_finite_nodata(domain_metadata["nodata"]):
        domain_valid &= domain != domain_metadata["nodata"]

    return domain_valid & (domain > 0)


def build_valid_probability_mask(
    probability: np.ndarray,
    nodata: float | int | None,
) -> np.ndarray:
    valid = np.isfinite(probability)
    if is_finite_nodata(nodata):
        valid &= probability != nodata
    return valid


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


def write_mask(
    output_path: Path,
    final_mask: np.ndarray,
    retained_domain: np.ndarray,
    profile: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = np.full(
        final_mask.shape,
        OUTPUT_NODATA,
        dtype=np.uint8,
    )
    output[retained_domain] = 0
    output[retained_domain & final_mask.astype(bool)] = 1

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


def main() -> None:
    args = parse_args()
    validate_parameters(args)

    probability, profile, metadata = read_probability(args.input)

    valid_probability = build_valid_probability_mask(
        probability,
        metadata["nodata"],
    )

    retained_domain = valid_probability.copy()
    if args.evaluation_domain is not None:
        retained_domain &= read_evaluation_domain(
            args.evaluation_domain,
            metadata,
        )

    otsu_sample = probability[
        retained_domain & (probability > args.min_probability)
    ]
    if otsu_sample.size == 0:
        raise RuntimeError(
            "No valid positive probabilities were found for Otsu estimation."
        )

    otsu_base_threshold = float(threshold_otsu(otsu_sample))
    high_threshold = otsu_base_threshold * args.high_multiplier
    low_threshold = otsu_base_threshold * args.low_multiplier

    probability_for_processing = np.where(
        valid_probability,
        probability,
        0.0,
    ).astype(np.float32)

    final_mask = postprocess_probability_map(
        probability_map=probability_for_processing,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        opening_radius=args.opening_radius,
        min_object_size=args.min_object_size,
        dilation_radius=args.dilation_radius,
    )

    final_mask = (
        final_mask.astype(bool) & retained_domain
    ).astype(np.uint8)

    write_mask(
        output_path=args.output,
        final_mask=final_mask,
        retained_domain=retained_domain,
        profile=profile,
    )

    print("Binary damage mask generated successfully.")
    print(f"Input probability map: {args.input}")
    print(
        "Evaluation domain: "
        + (
            str(args.evaluation_domain)
            if args.evaluation_domain is not None
            else "all valid probability pixels"
        )
    )
    print(f"Otsu base threshold: {otsu_base_threshold:.10f}")
    print(f"High multiplier: {args.high_multiplier:.2f}")
    print(f"Low multiplier: {args.low_multiplier:.2f}")
    print(f"Absolute high threshold: {high_threshold:.10f}")
    print(f"Absolute low threshold: {low_threshold:.10f}")
    print(f"Opening radius: {args.opening_radius}")
    print(f"Minimum object size: {args.min_object_size}")
    print(f"Dilation radius: {args.dilation_radius}")
    print(f"Retained-domain pixels: {int(retained_domain.sum())}")
    print(f"Damage pixels: {int(final_mask.sum())}")
    print(f"Output mask: {args.output}")


if __name__ == "__main__":
    main()
